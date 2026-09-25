"""
Distal-to-proximal actuator sizing for a 6- or 7-DOF serial arm.

Joints are clustered like most cobots:

    shoulder: yaw + pitch            (sized by shoulder pitch)
    elbow:    [upper-arm roll] + elbow pitch   (7-DOF adds the roll; sized by elbow pitch)
    wrist:    forearm roll + wrist pitch + wrist roll   (sized by wrist pitch)

Every joint in a cluster gets the same actuator. Sizing uses the static
gravity torque with the arm stretched out horizontally (link lengths from
analyze_arm_geometry), times a safety factor for acceleration:

    T_group = SF · g · [ m_payload · x_payload + Σ_distal m_i · x_i ]

where x is measured from the cluster's sizing (pitch) axis. The wrist cluster
carries its own distal roll actuator, so each cluster is solved by fixed-point
iteration; the outer loop repeats until every actuator mass has converged.
Structure mass (links) is a placeholder hook — `link_mass_per_m` — so a later
structure model can drop into the same loop.

Actuator = gearbox + motor, from the existing fits:

    gearbox:  log(m) = a[type] + b·log(T_out) + c·log(ratio)       (analyze_gearbox_mass)
    motor:    log(m) = a[form] + b·log(τ_motor) + c·log(ω_motor)   (analyze_motor_mass)
    τ_motor = T_group / (ratio · η[type]),  ω_motor = ω_joint · ratio

Actuator mass is multiplied by an integration factor calibrated on complete
joint modules (data/robot_joint_data.csv): actual module mass / (predicted bare
gearbox + predicted motor), median per gearbox type.

Usage:
    python arm_mass_model.py --reach 0.85 --payload 5
    python arm_mass_model.py --reach 1.3 --payload 10 --dof 7 --gearbox planetary --motor outrunner

Gearbox / motor "all" (the default) uses pooled fits over every type / form.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import analyze_arm_geometry as geo
import analyze_gearbox_mass as agb
import analyze_motor_mass as amm

G = 9.81
GROUPS = ("wrist", "elbow", "shoulder")  # sizing order: most distal first
GROUP_COLORS = {"shoulder": "#3987e5", "elbow": "#d95926", "wrist": "#199e70"}

# Categorical colors for arm Type (reference dark palette, slots 1–3 + 7)
ARM_TYPE_COLORS = {
    "collaborative": "#199e70",
    "industrial": "#d95926",
    "research": "#9085e9",
    "hobby": "#3987e5",
}
# Payload sweep: one curve per payload, reach on x. Each curve gets its own
# categorical hue (reference dark palette slots 1–5, fixed order) and published
# arms are colored by the curve whose payload they're closest to, so each dot
# can be read against its own line. Arm type moves to marker shape.
SWEEP_PAYLOADS = (1.0, 2.0, 5.0, 10.0, 20.0)
SWEEP_REACHES = tuple(np.round(np.linspace(0.25, 3.5, 66), 3))
SWEEP_COLORS = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")
OUT_OF_BAND_COLOR = "#5a6b7d"
ARM_TYPE_MARKERS = {"collaborative": "o", "industrial": "s", "research": "D", "hobby": "^"}


def payload_bin_edges(payloads: tuple[float, ...] = SWEEP_PAYLOADS) -> np.ndarray:
    """Geometric midpoints between curve payloads, plus one half-step past each end."""
    p = np.log(np.asarray(payloads, dtype=float))
    mids = (p[:-1] + p[1:]) / 2
    lo, hi = p[0] - (mids[0] - p[0]), p[-1] + (p[-1] - mids[-1])
    return np.exp(np.concatenate([[lo], mids, [hi]]))


def assign_payload_bin(payload_kg: pd.Series, payloads: tuple[float, ...] = SWEEP_PAYLOADS) -> pd.Series:
    """Nearest curve payload (log distance), or NaN outside the outer edges."""
    edges = payload_bin_edges(payloads)
    idx = np.digitize(payload_kg.to_numpy(dtype=float), edges) - 1
    out = np.full(len(idx), np.nan)
    ok = (idx >= 0) & (idx < len(payloads))
    out[ok] = np.asarray(payloads)[idx[ok]]
    return pd.Series(out, index=payload_kg.index)

# "all" = pooled fit over every gearbox type / motor form (no type/form term):
# the average design, used for the upper-bound curves.
ALL = "all"
GEARBOX_TYPES = [ALL, *agb.TYPE_COLORS]
MOTOR_FORMS = [ALL, *amm.FORM_COLORS]

# Typical efficiency at rated load (worm is ratio-dependent; 0.5 is mid-range).
GEAR_EFFICIENCY = {
    "harmonic": 0.75,
    "cycloidal": 0.85,
    "planetary": 0.90,
    "worm": 0.50,
    "spur": 0.90,
}
DEFAULT_RATIO = {
    "harmonic": 100.0,
    "cycloidal": 80.0,
    "planetary": 50.0,
    "worm": 50.0,
    "spur": 20.0,
}


def _pooled_gearbox_defaults() -> tuple[float, float]:
    """Efficiency and ratio for "all": weighted by how often each type appears in the data."""
    g = agb.load_gearboxes().dropna(subset=["Type", "Ratio"])
    counts = g["Type"].value_counts()
    eta = sum(GEAR_EFFICIENCY[t] * n for t, n in counts.items()) / counts.sum()
    return round(float(eta), 2), float(np.round(g["Ratio"].median()))


GEAR_EFFICIENCY[ALL], DEFAULT_RATIO[ALL] = _pooled_gearbox_defaults()

# Non-sizing joints (yaw / roll axes that don't carry the gravity moment) get this
# fraction of their cluster's sizing torque unless overridden per joint.
DEFAULT_SECONDARY_FRAC = 0.5
SECONDARY_JOINTS = ("Shoulder yaw", "Upper-arm roll", "Forearm roll", "Wrist roll")

# Joint-module Type → gearbox Type for calibrating the integration factor.
JOINT_TO_GEARBOX = {
    "harmonic": "harmonic",
    "cycloidal": "cycloidal",
    "planetary": "planetary",
    "qdd": "planetary",
}
# Module peak / rated torque when only one of the two is published.
PEAK_OVER_RATED = 2.5


@dataclass
class JointSlot:
    name: str
    group: str
    link: str  # link the actuator sits on: base / upper / forearm / wrist
    x: float  # horizontal distance from the shoulder pitch axis (m), arm stretched out
    sizing: bool = False


@dataclass
class ArmConfig:
    reach_m: float
    payload_kg: float
    dof: int = 6
    gearbox_type: str = ALL
    motor_form: str = ALL
    geometry: geo.GeometryFractions | None = None
    ratio: dict[str, float] = field(default_factory=dict)  # per group; defaults by gearbox type
    joint_speed_dps: dict[str, float] = field(
        default_factory=lambda: {"shoulder": 180.0, "elbow": 180.0, "wrist": 240.0}
    )
    safety_factor: float = 1.0
    # Joints off the torque chain (shoulder yaw, rolls) are sized to this fraction
    # of their cluster's sizing torque; keyed by joint name, missing → default.
    secondary_torque_frac: dict[str, float] = field(default_factory=dict)
    tool_offset_m: float = 0.0  # payload CoG beyond the flange
    integrated: bool = True  # apply joint-module calibration factor
    link_mass_per_m: float = 0.0  # structure placeholder (kg per m of link)
    # Actuator placement inside a cluster (fractions of the adjacent link)
    upper_roll_frac: float = 0.5  # 7-DOF upper-arm roll, along upper arm from shoulder
    forearm_roll_frac: float = 0.25  # forearm roll, back from the wrist toward the elbow
    wrist_roll_frac: float = 0.5  # distal wrist roll, out from wrist pitch toward flange

    def ratio_for(self, group: str) -> float:
        return float(self.ratio.get(group, DEFAULT_RATIO[self.gearbox_type]))

    def torque_frac(self, joint: "JointSlot") -> float:
        if joint.sizing:
            return 1.0
        return float(self.secondary_torque_frac.get(joint.name, DEFAULT_SECONDARY_FRAC))


@dataclass
class ActuatorModels:
    """Closed-form log-linear mass fits, keyed by gearbox type / motor form (+ "all").

    gear_coef[type]  = (a, b, c):  log m = a + b·log(T_out) + c·log(ratio)
    motor_coef[form] = (a, b, c):  log m = a + b·log(τ) + c·log(ω)
    Per-type/form entries share b, c from the one-hot fit; "all" is a separate
    pooled regression with its own slopes.
    """

    gear_coef: dict[str, tuple[float, float, float]]
    motor_coef: dict[str, tuple[float, float, float]]
    integration: dict[str, float]
    motor_ranges: pd.DataFrame  # per form (+ all): torque/speed min/max
    gear_ranges: pd.DataFrame  # per type (+ all): torque/ratio min/max
    motor_fit: dict
    gear_fit: dict

    def gearbox_mass(self, typ: str, torque_out: float, ratio: float) -> float:
        a, b, c = self.gear_coef[typ]
        return float(np.exp(a + b * np.log(torque_out) + c * np.log(ratio)))

    def motor_mass(self, form: str, torque: float, speed_rpm: float) -> float:
        a, b, c = self.motor_coef[form]
        return float(np.exp(a + b * np.log(torque) + c * np.log(speed_rpm)))


def _fit_pooled_gearbox(train: pd.DataFrame) -> tuple[float, float, float]:
    """log(m) = a + b·log(T_out) + c·log(ratio), no type term."""
    X = train[["log_torque", "log_ratio"]].to_numpy()
    lr = LinearRegression().fit(X, train["log_mass"].to_numpy())
    return float(lr.intercept_), float(lr.coef_[0]), float(lr.coef_[1])


def _calibrate_integration(gear_mass, motor_mass) -> dict[str, float]:
    """Median (module mass / predicted gearbox+motor) per gearbox type, plus "all"."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "robot_joint_data.csv")
    joints = pd.read_csv(path)
    ratios: dict[str, list[float]] = {}
    for _, r in joints.iterrows():
        gtype = JOINT_TO_GEARBOX.get(r["Type"])
        if gtype is None or pd.isna(r["Weight_kg"]) or pd.isna(r["Gear_Ratio"]):
            continue
        rated, peak = r["Rated_Torque_Nm"], r["Peak_Torque_Nm"]
        if pd.isna(rated) and pd.isna(peak):
            continue
        rated = peak / PEAK_OVER_RATED if pd.isna(rated) else rated
        peak = rated * PEAK_OVER_RATED if pd.isna(peak) else peak
        speed = 60.0 if pd.isna(r["Rated_Speed_rpm"]) else r["Rated_Speed_rpm"]
        ratio = float(r["Gear_Ratio"])
        for key, form in ((gtype, "frameless"), (ALL, ALL)):
            eta = GEAR_EFFICIENCY[key]
            m_pred = gear_mass(key, rated, ratio) + motor_mass(form, peak / (ratio * eta), speed * ratio)
            ratios.setdefault(key, []).append(float(r["Weight_kg"]) / m_pred)
    factors = {t: float(np.median(v)) for t, v in ratios.items() if len(v) >= 5}
    for t in GEARBOX_TYPES:
        factors.setdefault(t, 1.0)
    return factors


def _ranges(train: pd.DataFrame, by: str, **agg) -> pd.DataFrame:
    per = train.groupby(by).agg(**agg)
    per.loc[ALL] = train.assign(**{by: ALL}).groupby(by).agg(**agg).iloc[0]
    return per


@lru_cache(maxsize=1)
def load_actuator_models() -> ActuatorModels:
    m_train = amm.train_frame(amm.load_motors())
    m_fit = amm.fit_global(m_train)
    m_pool = amm.fit_pooled(m_train)
    g_train = agb.train_frame(agb.load_gearboxes())
    g_fit = agb.fit_global(g_train)

    gear_coef = {
        t: (
            g_fit["intercept"] + float(g_fit["coef"].get(f"type__Type_{t}", 0.0)),
            g_fit["b_torque"],
            g_fit["c_ratio"],
        )
        for t in g_train["Type"].unique()
    }
    gear_coef[ALL] = _fit_pooled_gearbox(g_train)
    motor_coef = {
        f: (a, m_fit["b_torque"], m_fit["c_speed"]) for f, a in amm.form_offsets(m_fit, m_train).items()
    }
    motor_coef[ALL] = (m_pool["a"], m_pool["b_torque"], m_pool["c_speed"])

    def gear_mass(t, T, i):
        a, b, c = gear_coef[t]
        return float(np.exp(a + b * np.log(T) + c * np.log(i)))

    def motor_mass(f, tau, w):
        a, b, c = motor_coef[f]
        return float(np.exp(a + b * np.log(tau) + c * np.log(w)))

    return ActuatorModels(
        gear_coef=gear_coef,
        motor_coef=motor_coef,
        integration=_calibrate_integration(gear_mass, motor_mass),
        motor_ranges=_ranges(
            m_train, "Form",
            t_min=("Max_Torque_Nm", "min"), t_max=("Max_Torque_Nm", "max"),
            w_min=("Max_Speed_rpm", "min"), w_max=("Max_Speed_rpm", "max"), n=("Weight_kg", "size"),
        ),
        gear_ranges=_ranges(
            g_train, "Type",
            t_min=("Rated_Output_Torque_Nm", "min"), t_max=("Rated_Output_Torque_Nm", "max"),
            r_min=("Ratio", "min"), r_max=("Ratio", "max"), n=("Weight_kg", "size"),
        ),
        motor_fit=m_fit,
        gear_fit=g_fit,
    )


@lru_cache(maxsize=None)
def default_geometry(basis: str = "all") -> geo.GeometryFractions:
    return geo.fit_fractions(geo.load_geometry(), geo.GEOMETRY_BASES[basis])


def layout_joints(cfg: ArmConfig) -> tuple[list[JointSlot], dict[str, float]]:
    """Joint chain (proximal → distal) with horizontal positions for the stretched-out pose."""
    L = (cfg.geometry or default_geometry()).lengths(cfg.reach_m)
    x_elbow = L["upper"]
    x_wrist = L["upper"] + L["forearm"]
    joints = [
        JointSlot("Shoulder yaw", "shoulder", "base", -L["offset"]),
        JointSlot("Shoulder pitch", "shoulder", "base", 0.0, sizing=True),
    ]
    if cfg.dof >= 7:
        joints.append(JointSlot("Upper-arm roll", "elbow", "upper", cfg.upper_roll_frac * L["upper"]))
    joints += [
        JointSlot("Elbow pitch", "elbow", "upper", x_elbow, sizing=True),
        JointSlot("Forearm roll", "wrist", "forearm", x_wrist - cfg.forearm_roll_frac * L["forearm"]),
        JointSlot("Wrist pitch", "wrist", "forearm", x_wrist, sizing=True),
        JointSlot("Wrist roll", "wrist", "wrist", x_wrist + cfg.wrist_roll_frac * L["wrist"]),
    ]
    L["x_elbow"] = x_elbow
    L["x_wrist"] = x_wrist
    L["x_flange"] = x_wrist + L["wrist"]
    L["x_payload"] = L["x_flange"] + cfg.tool_offset_m
    return joints, L


def _structure_masses(cfg: ArmConfig, L: dict[str, float]) -> list[tuple[str, float, float]]:
    """(link, mass, x of CoG) for the structure placeholder; links are uniform rods."""
    if cfg.link_mass_per_m <= 0:
        return []
    spans = {
        "upper": (0.0, L["x_elbow"]),
        "forearm": (L["x_elbow"], L["x_wrist"]),
        "wrist": (L["x_wrist"], L["x_flange"]),
    }
    return [(k, cfg.link_mass_per_m * (b - a), 0.5 * (a + b)) for k, (a, b) in spans.items()]


def size_actuator(
    cfg: ArmConfig, group: str, torque_nm: float, models: ActuatorModels
) -> dict[str, float]:
    ratio = cfg.ratio_for(group)
    eta = GEAR_EFFICIENCY[cfg.gearbox_type]
    motor_torque = torque_nm / (ratio * eta)
    motor_speed = cfg.joint_speed_dps[group] / 6.0 * ratio  # deg/s → rpm, then through the gear
    m_gear = models.gearbox_mass(cfg.gearbox_type, torque_nm, ratio)
    m_motor = models.motor_mass(cfg.motor_form, motor_torque, motor_speed)
    factor = models.integration[cfg.gearbox_type] if cfg.integrated else 1.0
    return {
        "ratio": ratio,
        "motor_torque_nm": motor_torque,
        "motor_speed_rpm": motor_speed,
        "gearbox_kg": m_gear * factor,
        "motor_kg": m_motor * factor,
        "actuator_kg": (m_gear + m_motor) * factor,
    }


def _range_warnings(cfg: ArmConfig, table: pd.DataFrame, models: ActuatorModels) -> list[str]:
    """Flag any actuator whose torque / ratio / motor point falls outside the fitted data."""
    warnings = []
    g = models.gear_ranges.loc[cfg.gearbox_type]
    m = models.motor_ranges.loc[cfg.motor_form]
    for _, s in table.iterrows():
        who = s["joint"].lower()
        if not g["t_min"] <= s["torque_nm"] <= g["t_max"]:
            warnings.append(
                f"{who}: {s['torque_nm']:.1f} Nm is outside the {cfg.gearbox_type} gearbox data "
                f"({g['t_min']:.3g}–{g['t_max']:.3g} Nm) — extrapolated"
            )
        if not g["r_min"] <= s["ratio"] <= g["r_max"]:
            warnings.append(
                f"{who}: ratio {s['ratio']:.0f} is outside {cfg.gearbox_type} data "
                f"({g['r_min']:.0f}–{g['r_max']:.0f})"
            )
        if not m["t_min"] <= s["motor_torque_nm"] <= m["t_max"]:
            warnings.append(
                f"{who}: motor {s['motor_torque_nm']:.3g} Nm is outside {cfg.motor_form} data "
                f"({m['t_min']:.3g}–{m['t_max']:.3g} Nm) — extrapolated"
            )
        if not m["w_min"] <= s["motor_speed_rpm"] <= m["w_max"]:
            warnings.append(
                f"{who}: motor {s['motor_speed_rpm']:.0f} rpm is outside {cfg.motor_form} data "
                f"({m['w_min']:.0f}–{m['w_max']:.0f} rpm)"
            )
    return warnings


def size_arm(
    cfg: ArmConfig,
    models: ActuatorModels | None = None,
    tol_kg: float = 1e-5,
    max_iter: int = 100,
) -> dict:
    """Size wrist → elbow → shoulder; iterate until actuator masses stop changing.

    The cluster's pitch joint takes the full gravity torque; the other joints in
    the cluster (yaw / rolls) take `cfg.torque_frac(joint)` of it.
    """
    models = models or load_actuator_models()
    joints, L = layout_joints(cfg)
    structure = _structure_masses(cfg, L)
    link_order = {"base": 0, "upper": 1, "forearm": 2, "wrist": 3}

    mass = {j.name: 0.0 for j in joints}
    sized: dict[str, dict] = {}
    per_joint: dict[str, dict] = {}
    history = []
    for it in range(max_iter):
        prev = dict(mass)
        for group in GROUPS:
            idx = next(i for i, j in enumerate(joints) if j.group == group and j.sizing)
            pivot = joints[idx]
            # Everything after the pivot in the chain is carried by it — including
            # same-cluster actuators (wrist roll on wrist pitch).
            distal = joints[idx + 1 :]
            moment = cfg.payload_kg * (L["x_payload"] - pivot.x)
            moment += sum(mass[j.name] * (j.x - pivot.x) for j in distal)
            for link, m_link, x_link in structure:
                if link_order[link] > link_order[pivot.link]:
                    moment += m_link * (x_link - pivot.x)
            static = G * moment
            torque = cfg.safety_factor * static
            act = size_actuator(cfg, group, torque, models)
            sized[group] = {
                "sizing_joint": pivot.name,
                "lever_m": L["x_payload"] - pivot.x,
                "static_torque_nm": static,
                "torque_nm": torque,
                **act,
            }
            for j in joints:
                if j.group != group:
                    continue
                frac = cfg.torque_frac(j)
                j_act = act if frac == 1.0 else size_actuator(cfg, group, torque * frac, models)
                per_joint[j.name] = {"torque_frac": frac, "torque_nm": torque * frac, **j_act}
                mass[j.name] = j_act["actuator_kg"]
        history.append({
            "iteration": it + 1,
            **{g: sized[g]["actuator_kg"] for g in GROUPS},
            "total": sum(mass.values()),
        })
        if max(abs(mass[k] - prev[k]) for k in mass) < tol_kg:
            break

    rows = []
    for j in joints:
        s = per_joint[j.name]
        rows.append(
            {
                "joint": j.name,
                "group": j.group,
                "sizing": j.sizing,
                "x_m": j.x,
                "torque_frac": s["torque_frac"],
                "torque_nm": s["torque_nm"],
                "ratio": s["ratio"],
                "motor_torque_nm": s["motor_torque_nm"],
                "motor_speed_rpm": s["motor_speed_rpm"],
                "motor_kg": s["motor_kg"],
                "gearbox_kg": s["gearbox_kg"],
                "actuator_kg": s["actuator_kg"],
            }
        )
    table = pd.DataFrame(rows)
    structure_kg = sum(m for _, m, _ in structure)
    return {
        "config": cfg,
        "lengths": L,
        "joints": joints,
        "table": table,
        "groups": sized,
        "history": pd.DataFrame(history),
        "iterations": len(history),
        "actuator_total_kg": float(table["actuator_kg"].sum()),
        "structure_kg": structure_kg,
        "structure": structure,
        "integration_factor": models.integration[cfg.gearbox_type] if cfg.integrated else 1.0,
        "warnings": _range_warnings(cfg, table, models),
    }


def payload_ratio_sweep(
    template: ArmConfig | None = None,
    payloads: tuple[float, ...] = SWEEP_PAYLOADS,
    reaches: tuple[float, ...] = SWEEP_REACHES,
    models: ActuatorModels | None = None,
    pooled: bool = True,
) -> pd.DataFrame:
    """Payload ratio (payload / actuator mass) across reach for each payload.

    With no structure mass this is an upper bound on what a real arm with the
    same actuators and geometry could reach. By default the gearbox and motor
    are the pooled "all" fits (average design, default ratio); the template
    still supplies DOF, geometry, SF, speeds and the structure placeholder.
    """
    models = models or load_actuator_models()
    template = template or ArmConfig(reach_m=1.0, payload_kg=1.0)
    if pooled:
        template = replace(template, gearbox_type=ALL, motor_form=ALL, ratio={})
    rows = []
    for payload in payloads:
        for reach in reaches:
            res = size_arm(replace(template, reach_m=float(reach), payload_kg=float(payload)), models)
            arm_kg = res["actuator_total_kg"] + res["structure_kg"]
            rows.append(
                {
                    "payload_kg": payload,
                    "reach_m": reach,
                    "actuator_kg": res["actuator_total_kg"],
                    "arm_kg": arm_kg,
                    "payload_ratio": payload / arm_kg,
                }
            )
    return pd.DataFrame(rows)


def load_arm_catalog() -> pd.DataFrame:
    """Published arms with reach, payload and mass; adds Payload_Ratio."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "robot_arm_data.csv")
    df = pd.read_csv(path).dropna(subset=["Payload_kg", "Reach_m", "Weight_kg"])
    df = df[(df["Weight_kg"] > 0) & (df["Payload_kg"] > 0) & (df["Reach_m"] > 0)].copy()
    df["Payload_Ratio"] = df["Payload_kg"] / df["Weight_kg"]
    return df


def plot_payload_ratio_bounds(sweep: pd.DataFrame, arms: pd.DataFrame, out: str, title_note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#0a1628")
    geo._style(ax)
    arms = arms.assign(bin=assign_payload_bin(arms["Payload_kg"]))
    edges = payload_bin_edges()

    # Out-of-band arms first, dimmed, so the comparable ones sit on top
    far = arms[arms["bin"].isna()]
    for typ, marker in ARM_TYPE_MARKERS.items():
        d = far[far["Type"] == typ]
        ax.scatter(d["Reach_m"], d["Payload_Ratio"], s=18, c=OUT_OF_BAND_COLOR, marker=marker,
                   alpha=0.35, linewidths=0, zorder=2)
    handles = []
    for i, (payload, color) in enumerate(zip(SWEEP_PAYLOADS, SWEEP_COLORS)):
        near = arms[arms["bin"] == payload]
        for typ, marker in ARM_TYPE_MARKERS.items():
            d = near[near["Type"] == typ]
            ax.scatter(d["Reach_m"], d["Payload_Ratio"], s=38, c=color, marker=marker, alpha=0.9,
                       edgecolors="#0a1628", linewidths=0.8, zorder=3)
        c = sweep[sweep["payload_kg"] == payload]
        ax.plot(c["reach_m"], c["payload_ratio"], color="#0a1628", linewidth=4.5, zorder=4)  # surface halo
        (line,) = ax.plot(c["reach_m"], c["payload_ratio"], color=color, linewidth=2.2, zorder=5,
                          label=f"{payload:g} kg  (arms {edges[i]:.2g}–{edges[i + 1]:.2g} kg, n={len(near)})")
        handles.append(line)
        end = c.iloc[-1]
        ax.annotate(f"{payload:g} kg", (end["reach_m"], end["payload_ratio"]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#e8f4fc")
    from matplotlib.lines import Line2D

    handles += [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="#a9b8c8", markeredgecolor="none",
               markersize=7, label=t)
        for t, m in ARM_TYPE_MARKERS.items()
    ]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=OUT_OF_BAND_COLOR, alpha=0.5,
                          markeredgecolor="none", markersize=6,
                          label=f"payload outside {edges[0]:.2g}–{edges[-1]:.2g} kg (n={len(far)})"))
    ax.set_yscale("log")
    ax.set_xlim(0, 4.8)
    ax.set_ylim(0.01, max(2.0, float(sweep["payload_ratio"].max()) * 1.4))
    ax.set_xlabel("Reach (m)")
    ax.set_ylabel("Payload ratio (payload / arm mass)")
    ax.set_title("Payload ratio vs reach — curves are actuator-only upper bounds; dots colored by nearest curve payload"
                 + title_note)
    leg = ax.legend(handles=handles, loc="upper right", fontsize=8.5, facecolor="#121f36", edgecolor="#2a4060",
                    labelcolor="#e8f4fc", title="bound curve = arm payload band  ·  marker = arm type",
                    title_fontsize=8.5)
    leg.get_title().set_color("#6b8ba4")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


def validate_against_arms(cfg_template: ArmConfig | None = None) -> pd.DataFrame:
    """Run the actuator-only model on arms with published joint torques / masses."""
    df = geo.load_geometry()
    rows = []
    for _, r in df.iterrows():
        frac = geo.GeometryFractions(
            base=r["frac_base"],
            offset=r["frac_offset"],
            upper=r["frac_upper"],
            forearm=r["frac_forearm"],
            wrist=r["frac_wrist"],
        )
        base = cfg_template or ArmConfig(
            reach_m=1.0, payload_kg=1.0, gearbox_type="harmonic", motor_form="frameless"
        )
        cfg = replace(
            base,
            reach_m=float(r["Chain_m"]),
            payload_kg=float(r["Payload_kg"]),
            dof=int(r["DOF"]),
            geometry=frac,
        )
        res = size_arm(cfg)
        out = {
            "Name": r["Name"],
            "Type": r["Type"],
            "DOF": int(r["DOF"]),
            "Reach_m": r["Reach_m"],
            "Payload_kg": r["Payload_kg"],
            "Weight_kg": r["Weight_kg"],
            "actuator_kg": res["actuator_total_kg"],
        }
        for g in GROUPS:
            out[f"{g}_pred_nm"] = res["groups"][g]["torque_nm"]
            out[f"{g}_actual_nm"] = r[f"{g.capitalize()}_Torque_Nm"]
        rows.append(out)
    return pd.DataFrame(rows)


def plot_validation(val: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#0a1628")

    ax = axes[0]
    geo._style(ax)
    for g in GROUPS:
        d = val.dropna(subset=[f"{g}_actual_nm"])
        ax.scatter(
            d[f"{g}_actual_nm"],
            d[f"{g}_pred_nm"],
            c=GROUP_COLORS[g],
            s=55,
            edgecolors="white",
            linewidths=0.4,
            label=g,
            zorder=3,
        )
        for _, r in d.iterrows():
            if g == "shoulder":
                ax.annotate(r["Name"], (r[f"{g}_actual_nm"], r[f"{g}_pred_nm"]), fontsize=6,
                            color="#6b8ba4", xytext=(3, -8), textcoords="offset points")
    lo, hi = 3, 1500
    ax.plot([lo, hi], [lo, hi], color="#6b8ba4", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Published joint torque rating (Nm)")
    ax.set_ylabel("Model torque, actuators + payload only (Nm)")
    ax.set_title("Torque: gap below the diagonal = structure + dynamics not yet modeled")
    ax.legend(fontsize=8, facecolor="#121f36", edgecolor="#2a4060", labelcolor="#e8f4fc")

    ax = axes[1]
    geo._style(ax)
    for typ, color in ARM_TYPE_COLORS.items():
        d = val[val["Type"] == typ].dropna(subset=["Weight_kg"])
        if d.empty:
            continue
        ax.scatter(d["Weight_kg"], d["actuator_kg"], c=color, s=55, edgecolors="white",
                   linewidths=0.4, label=typ, zorder=3)
    frac = (val["actuator_kg"] / val["Weight_kg"]).dropna()
    xs = np.logspace(-0.3, 3.2, 20)
    ax.plot(xs, xs, color="#6b8ba4", linestyle="--", linewidth=1)
    ax.plot(xs, xs * frac.median(), color="#00d4ff", linestyle=":", linewidth=1.2,
            label=f"median actuator share {frac.median():.0%}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Published robot mass (kg)")
    ax.set_ylabel("Model actuator mass (kg)")
    ax.set_title(f"Actuator-only mass vs whole-robot mass (off-chain joints at {DEFAULT_SECONDARY_FRAC:.0%})")
    ax.legend(fontsize=8, facecolor="#121f36", edgecolor="#2a4060", labelcolor="#e8f4fc")

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


def print_result(res: dict) -> None:
    cfg = res["config"]
    L = res["lengths"]
    print(
        f"\nArm: reach {cfg.reach_m:.2f} m  payload {cfg.payload_kg:.2f} kg  {cfg.dof}-DOF  "
        f"{cfg.gearbox_type} + {cfg.motor_form}  SF={cfg.safety_factor}"
    )
    print(
        f"Links: base {L['base']:.3f}  offset {L['offset']:.3f}  upper {L['upper']:.3f}  "
        f"forearm {L['forearm']:.3f}  wrist {L['wrist']:.3f} m"
    )
    print(f"Converged in {res['iterations']} iterations; integration factor {res['integration_factor']:.2f}")
    t = res["table"].copy()
    print(
        t.to_string(
            index=False,
            formatters={
                "x_m": "{:.3f}".format,
                "torque_frac": "{:.0%}".format,
                "torque_nm": "{:.1f}".format,
                "ratio": "{:.0f}".format,
                "motor_torque_nm": "{:.3f}".format,
                "motor_speed_rpm": "{:.0f}".format,
                "motor_kg": "{:.3f}".format,
                "gearbox_kg": "{:.3f}".format,
                "actuator_kg": "{:.3f}".format,
            },
        )
    )
    print(f"Total actuator mass: {res['actuator_total_kg']:.2f} kg")
    for w in res["warnings"]:
        print(f"  ! {w}")


def main():
    parser = argparse.ArgumentParser(description="Distal-to-proximal actuator mass estimate")
    parser.add_argument("--reach", type=float, default=0.85)
    parser.add_argument("--payload", type=float, default=5.0)
    parser.add_argument("--dof", type=int, default=6, choices=[6, 7])
    parser.add_argument("--gearbox", default=ALL, choices=GEARBOX_TYPES)
    parser.add_argument("--motor", default=ALL, choices=MOTOR_FORMS)
    parser.add_argument("--geometry", default="all", choices=list(geo.GEOMETRY_BASES))
    parser.add_argument("--sf", type=float, default=1.0, help="Safety factor on static torque")
    parser.add_argument(
        "--secondary", type=float, default=DEFAULT_SECONDARY_FRAC,
        help="Torque fraction for joints off the torque chain (yaw / rolls), 0–1",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_arm_actuator_validation.png"),
    )
    parser.add_argument(
        "--out-bounds",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_arm_payload_ratio_bounds.png"),
    )
    args = parser.parse_args()

    models = load_actuator_models()
    print("Integration factors (module mass / bare gearbox+motor):")
    for t, f in models.integration.items():
        print(f"  {t:10s} {f:.2f}")

    cfg = ArmConfig(
        reach_m=args.reach,
        payload_kg=args.payload,
        dof=args.dof,
        gearbox_type=args.gearbox,
        motor_form=args.motor,
        geometry=default_geometry(args.geometry),
        safety_factor=args.sf,
        secondary_torque_frac={j: args.secondary for j in SECONDARY_JOINTS},
    )
    print_result(size_arm(cfg, models))

    sweep = payload_ratio_sweep(replace(cfg, reach_m=1.0, payload_kg=1.0), models=models)
    print(f"\nPayload-ratio upper bound ({args.dof}-DOF, all gearboxes + all motors pooled, no structure):")
    pivot = sweep.pivot(index="reach_m", columns="payload_kg", values="payload_ratio")
    print(pivot.loc[[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]].to_string(float_format=lambda v: f"{v:.2f}"))
    plot_payload_ratio_bounds(
        sweep, load_arm_catalog(), args.out_bounds,
        f"\n{args.dof}-DOF, pooled gearbox + motor fits (all types), SF {args.sf}, "
        f"off-chain joints (yaw / rolls) at {args.secondary:.0%} of pitch torque",
    )

    val = validate_against_arms()
    print(f"\nValidation (harmonic + frameless, SF 1.0, off-chain joints {DEFAULT_SECONDARY_FRAC:.0%}, each arm's own geometry):")
    cols = ["Name", "Payload_kg", "Weight_kg", "actuator_kg"] + [
        f"{g}_{k}" for g in ["shoulder", "elbow", "wrist"] for k in ["pred_nm", "actual_nm"]
    ]
    print(val[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    plot_validation(val, args.out)


if __name__ == "__main__":
    main()
