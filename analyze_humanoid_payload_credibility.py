"""
What does a humanoid "payload" number actually mean? Four checks that lean on
the motor / gearbox / joint-module data and the arm mass model.

1) humanoid_peak_vs_rated.png
   Peak ÷ rated torque across our motors, joint modules and gearboxes, with the
   few humanoids that publish both a peak and a sustained payload drawn on the
   same axis. The joint-module median is the "peak factor" used below, and it
   turns a peak-only claim into an implied sustained rating.

2) humanoid_payload_plausibility.png
   For every humanoid with a whole-body payload, size one arm with
   arm_mass_model so its actuators use a typical humanoid share of body mass
   (quartiles of analyze_humanoid_scaling.arm_actuator_shares), then ask what
   two such arms can hold: sustained at full reach, sustained close to the body
   (25 cm), and at peak torque close to the body. The published number is
   placed against those three.

3) humanoid_capacity_vs_reach.png
   Two-arm sustained capacity vs horizontal hand distance from the shoulder for
   a few humanoids, next to the NIOSH recommended weight limit (23 kg × 25 cm / H,
   all other multipliers 1) and single-arm cobots from the arm catalog.

4) humanoid_holding_power.png
   Rough battery cost of holding a load. Motor constant Km is estimated from
   each motor's continuous torque and its surface area at an assumed heat flux
   (no winding resistance in the data), fit vs motor mass, then used for the
   copper loss of the arm's pitch joints while holding. Runtime =
   battery energy / (published average power + holding power).

Arm pose for capacity at hand distance h (from the shoulder pitch axis): forearm
horizontal, elbow at max(0, h − forearm − wrist), upper arm hanging/angled, so at
full reach this reduces exactly to the sized (stretched-out) case. Shoulder
and elbow pitch each limit the payload (the smaller wins); wrist pitch is left out
because a two-handed carry clamps the load close to the wrist axis.

Usage:
    python analyze_humanoid_payload_credibility.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import analyze_humanoid_scaling as hs
import arm_mass_model as amodel
from analyze_humanoid_scaling import (
    BG, BOUND_COLOR, GRID, HUMAN_COLOR, HUMANOID_COLOR, INK, MUTED, PANEL, style,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
MOTOR_PATH = os.path.join(script_dir, 'data', 'bldc_motor_data.csv')
JOINT_PATH = os.path.join(script_dir, 'data', 'robot_joint_data.csv')
GEARBOX_PATH = os.path.join(script_dir, 'data', 'gearbox_data.csv')
OUT_PEAK = os.path.join(script_dir, 'humanoid_peak_vs_rated.png')
OUT_PLAUSIBLE = os.path.join(script_dir, 'humanoid_payload_plausibility.png')
OUT_REACH = os.path.join(script_dir, 'humanoid_capacity_vs_reach.png')
OUT_POWER = os.path.join(script_dir, 'humanoid_holding_power.png')

G = amodel.G
CLOSE_M = 0.25            # NIOSH ideal horizontal distance
NIOSH_LC_KG = 23.0        # NIOSH load constant
NIOSH_H_MAX_M = 0.63      # HM = 0 beyond 63 cm
MAX_ARM_REACH_M = 1.2     # larger Arm_Reach_m values are workspace reach (Atlas 2.3 m)
PASS_COLOR = '#2ecc71'
PEAK_COLOR = '#f1c40f'
FAIL_COLOR = '#e74c3c'

# Humanoids that publish both a sustained/rated and a peak payload
# (values from the Notes column of data/humanoid_robot_data.csv).
HUMANOID_PEAK = {
    'Atlas (Electric)': (30.0, 50.0, 'whole body'),
    'Foundation Phantom MK-1': (20.0, 36.0, 'whole body'),
    'Unitree H2': (7.0, 15.0, 'per arm'),
    'Unitree H1-2': (7.0, 21.0, 'per arm'),
}
# Payloads published only as a claim / maximum with no duty cycle
# (Wandercraft Calvin-40's 40 kg is described as repetitive, so it isn't here)
PEAK_ONLY_CLAIMS = {'AgiBot A2-Max': 40.0}

# Battery energy (kWh) from the Notes column; only robots that also publish runtime + payload
BATTERY_KWH = {'Unitree H2': 0.972, 'Figure 03': 2.3, 'Kepler K2 Bumblebee': 2.33}
# Continuous-rating heat flux off the motor's outer surface (assumed, not in the data):
# ~25–30 W/m²K × ~70 K rise. Band is ×0.5 / ×2.
HEAT_FLUX_W_M2 = 2000.0
HEAT_FLUX_BAND = (0.5, 2.0)

CAPACITY_ROBOTS = ['Unitree G1', 'Unitree H2', 'Kepler K2 Bumblebee',
                   'Atlas (Electric)', 'Digit v5', 'Figure 03']
CAPACITY_COLORS = ['#3498db', '#9b59b6', '#e67e22', '#00d4ff', '#2ecc71', '#e84393']


# ---------------------------------------------------------------------------
# Shared: arm sizing and capacity in a bent pose
# ---------------------------------------------------------------------------

def arm_reach(r: pd.Series) -> tuple[float | None, bool]:
    """Shoulder→hand reach; (value, published?)."""
    if pd.notna(r['Arm_Reach_m']) and r['Arm_Reach_m'] <= MAX_ARM_REACH_M:
        return float(r['Arm_Reach_m']), True
    if pd.notna(r['Height_m']):
        return hs.ARM_REACH_OVER_HEIGHT * float(r['Height_m']), False
    return None, False


def arm_dof(r: pd.Series) -> int:
    # Same rule as analyze_humanoid_scaling: the model supports 6 or 7 DOF
    return 7 if pd.isna(r['Arm_DOF']) or r['Arm_DOF'] >= 7 else 6


def pose(res: dict, h: float) -> tuple[float, dict[str, float]]:
    """Elbow x and every elbow/wrist-cluster actuator's x with the hand at h, forearm horizontal."""
    L = res['lengths']
    lf = L['x_payload'] - L['x_elbow']
    e = max(0.0, h - lf)
    scale = (h - e) / lf
    xs = {}
    for j in res['joints']:
        if j.group == 'shoulder':
            continue
        if j.link == 'upper':
            xs[j.name] = j.x / L['x_elbow'] * e
        else:
            xs[j.name] = e + (j.x - L['x_elbow']) * scale
    return e, xs


def joint_torques(res: dict, payload: float, h: float) -> dict[str, float]:
    """Static gravity torque (N·m) on shoulder and elbow pitch holding `payload` at h.

    Wrist pitch is left out: in a two-handed carry the load is clamped between the
    palms, close to the wrist axis, so its lever doesn't shrink as the load comes in
    the way the model's stretched-out flange lever would suggest.
    """
    m = dict(zip(res['table']['joint'], res['table']['actuator_kg']))
    e, xs = pose(res, h)
    wrist = [j.name for j in res['joints'] if j.group == 'wrist']
    return {
        'shoulder': G * (payload * h + sum(m[k] * x for k, x in xs.items())),
        'elbow': G * (payload * (h - e) + sum(m[k] * (xs[k] - e) for k in wrist)),
    }


def arm_capacity(res: dict, h: float, torque_scale: float = 1.0) -> float:
    """Largest payload one arm holds at hand distance h with joint torques × torque_scale."""
    empty = joint_torques(res, 0.0, h)
    rated = {g: res['groups'][g]['torque_nm'] * torque_scale for g in empty}
    unit = {g: joint_torques(res, 1.0, h)[g] - empty[g] for g in empty}
    caps = [(rated[g] - empty[g]) / unit[g] for g in empty if unit[g] > 0]
    return max(0.0, min(caps))


def size_for_budget(reach: float, dof: int, budget_kg: float, models) -> dict:
    """Arm whose total actuator mass equals budget_kg (bisection on payload)."""
    lo, hi = 0.0, 200.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        res = amodel.size_arm(amodel.ArmConfig(reach, max(mid, 1e-3), dof=dof), models)
        if res['actuator_total_kg'] > budget_kg:
            hi = mid
        else:
            lo = mid
    return amodel.size_arm(amodel.ArmConfig(reach, max(lo, 1e-3), dof=dof), models)


# ---------------------------------------------------------------------------
# 1) Peak ÷ rated
# ---------------------------------------------------------------------------

def peak_ratios() -> dict[str, np.ndarray]:
    m = pd.read_csv(MOTOR_PATH).dropna(subset=['Max_Torque_Nm', 'Cont_Torque_Nm'])
    j = pd.read_csv(JOINT_PATH).dropna(subset=['Rated_Torque_Nm', 'Peak_Torque_Nm'])
    g = pd.read_csv(GEARBOX_PATH).dropna(subset=['Rated_Output_Torque_Nm', 'Peak_Output_Torque_Nm'])
    groups = {'BLDC motors (peak ÷ continuous)': (m['Max_Torque_Nm'] / m['Cont_Torque_Nm']).to_numpy()}
    jr = j['Peak_Torque_Nm'] / j['Rated_Torque_Nm']
    groups['Joint modules, all'] = jr.to_numpy()
    for typ in sorted(j['Type'].dropna().unique()):
        groups[f'  joint: {typ}'] = jr[j['Type'] == typ].to_numpy()
    groups['Gearboxes (peak ÷ rated output)'] = (
        g['Peak_Output_Torque_Nm'] / g['Rated_Output_Torque_Nm']).to_numpy()
    return groups


def plot_peak(groups: dict[str, np.ndarray], out: str) -> float:
    joint_med = float(np.median(groups['Joint modules, all']))
    jq = np.quantile(groups['Joint modules, all'], [0.25, 0.75])
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(BG)
    style(ax)
    names = list(groups)
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        v = groups[name]
        y = len(names) - 1 - i
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=14, color=BOUND_COLOR, alpha=0.45,
                   edgecolors='none', zorder=2)
        q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        ax.plot([q1, q3], [y, y], color=INK, linewidth=5, alpha=0.5, solid_capstyle='round', zorder=3)
        ax.plot([med], [y], marker='|', markersize=18, color='white', markeredgewidth=2.5, zorder=4)
        ax.text(5.15, y, f'median {med:.2f}  (n={len(v)})', color=MUTED, fontsize=9, va='center')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], color=INK, fontsize=10)

    top = len(names) - 0.4
    for k, (name, (rated, peak, basis)) in enumerate(sorted(HUMANOID_PEAK.items(),
                                                             key=lambda kv: kv[1][1] / kv[1][0])):
        ratio = peak / rated
        ax.axvline(ratio, color=HUMANOID_COLOR, linestyle='--', linewidth=1.4, alpha=0.9, zorder=1)
        ax.text(ratio + 0.03, top + 0.3 + 0.62 * k, f'{name}: {peak:g}/{rated:g} kg {basis} = {ratio:.1f}×',
                color=HUMANOID_COLOR, fontsize=8.5, ha='left', va='bottom', fontweight='bold',
                bbox=dict(facecolor=BG, edgecolor='none', pad=1))

    lines = [f'Joint-module peak factor: median {joint_med:.2f}× (IQR {jq[0]:.2f}–{jq[1]:.2f}×)',
             'Implied sustained rating for peak-only claims:']
    for name, claim in PEAK_ONLY_CLAIMS.items():
        lines.append(f'  {name}: {claim:g} kg → {claim / joint_med:.0f} kg '
                     f'({claim / jq[1]:.0f}–{claim / jq[0]:.0f})')
    ax.text(0.99, 0.03, '\n'.join(lines), transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9.5, color=INK, bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle='round,pad=0.5'),
            zorder=6)

    ax.set_xlim(0.8, 6.6)
    ax.set_ylim(-0.6, len(names) + 2.3)
    ax.set_xlabel('Peak ÷ rated (continuous) torque')
    ax.set_title('Atlas and Phantom quote peak at ~1.7–1.8× sustained; Unitree at 2–3×, like QDD joints',
                 fontsize=14, fontweight='bold', pad=10)
    fig.text(0.01, 0.005, 'Bars: interquartile range, white tick: median.  Humanoid lines: published peak '
             'payload ÷ published sustained / rated payload.', color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')
    return joint_med


# ---------------------------------------------------------------------------
# 2) Published payload vs what a typical-mass arm pair can hold
# ---------------------------------------------------------------------------

def plausibility(hum: pd.DataFrame, budgets: tuple[float, float, float], peak: float, models) -> pd.DataFrame:
    rows = []
    d = hum.dropna(subset=['Payload_kg', 'Weight_kg'])
    d = d[~d['Name'].isin(hs.HUMANOID_NOT_CARRY)]
    for _, r in d.iterrows():
        reach, listed = arm_reach(r)
        if reach is None:
            continue
        caps = {}
        for tag, frac in zip(('lo', 'mid', 'hi'), budgets):
            res = size_for_budget(reach, arm_dof(r), frac * float(r['Weight_kg']), models)
            h_full = res['lengths']['x_payload']
            caps[tag] = {
                'full': 2 * arm_capacity(res, h_full),
                'close': 2 * arm_capacity(res, CLOSE_M),
                'close_peak': 2 * arm_capacity(res, CLOSE_M, peak),
            }
        p = float(r['Payload_kg'])
        mid = caps['mid']
        # Where Payload_kg is the published peak, judge it against peak capacity only
        rated, peak_kg, basis = HUMANOID_PEAK.get(r['Name'], (np.nan, np.nan, ''))
        is_peak = basis == 'whole body' and p == peak_kg
        if is_peak:
            verdict = 'peak' if p <= mid['close_peak'] else 'beyond'
        else:
            verdict = 'sustained' if p <= mid['close'] else 'peak' if p <= mid['close_peak'] else 'beyond'
        rows.append({
            'Name': r['Name'], 'Research': r.get('Category') == 'research', 'Payload_kg': p, 'Body_kg': float(r['Weight_kg']), 'Reach_m': reach,
            'Reach_listed': listed,
            **{f'{k}_{t}': caps[t][k] for t in caps for k in caps[t]},
            'Sustained_kg': rated if is_peak else np.nan,
            'Verdict': verdict,
        })
    # Ascending so the highest full-reach capacity plots at the top
    return pd.DataFrame(rows).sort_values('full_mid').reset_index(drop=True)


def plot_plausibility(df: pd.DataFrame, budgets, peak: float, out: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 0.38 * len(df) + 2.6))
    fig.patch.set_facecolor(BG)
    style(ax)
    y = np.arange(len(df))
    color = {'sustained': PASS_COLOR, 'peak': PEAK_COLOR, 'beyond': FAIL_COLOR}
    for i, r in df.iterrows():
        ax.plot([r['full_mid'], r['close_peak_mid']], [i, i], color=GRID, linewidth=6,
                solid_capstyle='round', zorder=1)
        ax.plot([r['close_lo'], r['close_hi']], [i, i], color=BOUND_COLOR, linewidth=2, alpha=0.8, zorder=2)
    ax.scatter(df['full_mid'], y, s=40, facecolors=BG, edgecolors=BOUND_COLOR, linewidths=1.5, zorder=3)
    ax.scatter(df['close_mid'], y, s=55, color=BOUND_COLOR, edgecolors='white', linewidths=0.6, zorder=3)
    ax.scatter(df['close_peak_mid'], y, s=55, marker='s', facecolors=BG, edgecolors=PEAK_COLOR,
               linewidths=1.5, zorder=3)
    sus = df['Sustained_kg'].notna()
    ax.scatter(df.loc[sus, 'Sustained_kg'], y[sus], s=110, marker='D', facecolors=BG, edgecolors='white',
               linewidths=1.4, zorder=4)
    for i in np.flatnonzero(sus):
        r = df.iloc[i]
        ax.annotate(f"{r['Sustained_kg']:g} kg sustained / {r['Payload_kg']:g} kg instantaneous",
                    (r['Sustained_kg'], i), xytext=(0, 9), textcoords='offset points', ha='center',
                    fontsize=8, color=INK, zorder=6)
    ax.scatter(df['Payload_kg'], y, s=110, marker='D', c=[color[v] for v in df['Verdict']],
               edgecolors='white', linewidths=0.8, zorder=4)
    ax.axvline(NIOSH_LC_KG, color=HUMAN_COLOR, linestyle=':', linewidth=1.3, zorder=0)
    ax.text(NIOSH_LC_KG, 1.003, 'NIOSH 23 kg', color=HUMAN_COLOR, fontsize=9, ha='center', va='bottom',
            transform=ax.get_xaxis_transform())

    ax.set_xscale('log')
    ticks = [t for t in (2, 3, 5, 10, 20, 30, 50, 100) if df['close_lo'].min() * 0.5 <= t]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([n + ('' if listed else '*') for n, listed in zip(df['Name'], df['Reach_listed'])],
                       color=INK, fontsize=9.5)
    hs.color_research_ticks(ax, df['Research'])
    ax.set_ylim(-0.7, len(df) + 0.2)
    ax.set_xlabel('Two-arm payload (kg, log)')
    counts = df['Verdict'].value_counts()
    ax.set_title(f'Published payloads vs two typical-mass arms: {counts.get("sustained", 0)} fit sustained, '
                 f'{counts.get("peak", 0)} need peak torque, {counts.get("beyond", 0)} exceed both',
                 fontsize=13.5, fontweight='bold', pad=22)
    lo, mid, hi = (b * 100 for b in budgets)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='none', markerfacecolor=BG, markeredgecolor=BOUND_COLOR,
               markersize=7, label='Model: sustained at full reach'),
        Line2D([0], [0], marker='o', color=BOUND_COLOR, markerfacecolor=BOUND_COLOR, markeredgecolor='white',
               markersize=8, label=f'Model: sustained at {CLOSE_M * 100:.0f} cm (line: {lo:.1f}–{hi:.1f} % budget)'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=BG, markeredgecolor=PEAK_COLOR,
               markersize=8, label=f'Model: peak ({peak:.2f}× rated) at {CLOSE_M * 100:.0f} cm'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=PASS_COLOR, markeredgecolor='white',
               markersize=8, label='Published ≤ sustained close-in'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=PEAK_COLOR, markeredgecolor='white',
               markersize=8, label='Published needs peak torque'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=FAIL_COLOR, markeredgecolor='white',
               markersize=8, label='Published beyond modeled arms'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=BG, markeredgecolor='white',
               markersize=8, label='Published sustained, when the stored payload is a peak'),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.045), ncol=3, fontsize=8.5, facecolor=PANEL, edgecolor=GRID,
        labelcolor=INK, framealpha=0.95)
    fig.text(0.01, 0.005,
             f'Each arm sized so its actuators are {mid:.1f} % of body mass (median of humanoids with per-arm '
             f'ratings; band = quartiles). Pooled gearbox + motor fits, SF 1.0.  Red = the load must ride on the '
             f'torso / hips or the arms are heavier than typical.  * reach not published — 0.44 × height.  '
             f'Pink names = research / open-source humanoids.',
             color=MUTED, fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


# ---------------------------------------------------------------------------
# 3) Capacity vs hand distance, against NIOSH
# ---------------------------------------------------------------------------

def capacity_arm(r: pd.Series, budget_frac: float, models) -> tuple[dict, str]:
    reach, _ = arm_reach(r)
    if pd.notna(r['Arm_Payload_kg']):
        res = amodel.size_arm(amodel.ArmConfig(reach, float(r['Arm_Payload_kg']), dof=arm_dof(r)), models)
        return res, f"rated {r['Arm_Payload_kg']:g} kg/arm"
    res = size_for_budget(reach, arm_dof(r), budget_frac * float(r['Weight_kg']), models)
    return res, f'{budget_frac * 100:.1f} % arm budget'


def plot_reach(hum: pd.DataFrame, arms: pd.DataFrame, budget_frac: float, models, out: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    style(ax)
    cob = arms[(arms['Type'] == 'collaborative') & (arms['Reach_m'] <= 1.0)]
    ax.scatter(cob['Reach_m'], cob['Payload_kg'], s=16, color=hs.ARM_TYPE_COLORS['collaborative'],
               alpha=0.35, edgecolors='none', label=f'Cobot, single arm at full reach ({len(cob)})', zorder=1)

    H = np.linspace(0.10, NIOSH_H_MAX_M, 100)
    ax.plot(H, NIOSH_LC_KG * np.minimum(1.0, CLOSE_M / H), color=HUMAN_COLOR, linewidth=3,
            label='NIOSH recommended limit, ideal lift (23 kg × 25 cm / H)', zorder=4)
    ax.plot([NIOSH_H_MAX_M, NIOSH_H_MAX_M], [NIOSH_LC_KG * CLOSE_M / NIOSH_H_MAX_M, 0.3],
            color=HUMAN_COLOR, linewidth=1.2, linestyle=':', zorder=4)

    by = hum.set_index('Name')
    for name, color in zip(CAPACITY_ROBOTS, CAPACITY_COLORS):
        r = by.loc[name]
        res, basis = capacity_arm(r, budget_frac, models)
        h_max = res['lengths']['x_payload']
        hs_ = np.linspace(0.10, h_max, 60)
        cap = [2 * arm_capacity(res, h) for h in hs_]
        ax.plot(hs_, cap, color=color, linewidth=2.2, label=f'{name} — 2 arms, {basis}', zorder=3)
        if pd.notna(r['Payload_kg']):
            ax.scatter([CLOSE_M], [r['Payload_kg']], marker='D', s=70, color=color, edgecolors='white',
                       linewidths=0.8, zorder=5)
    ax.text(CLOSE_M + 0.008, 160, '◆ = published payload, placed at 25 cm', color=MUTED, fontsize=8.5, va='top')
    ax.set_yscale('log')
    ax.set_xlim(0.08, 1.0)
    ax.set_ylim(0.3, 200)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('Horizontal hand distance from shoulder (m) — NIOSH H from ankles for the human')
    ax.set_ylabel('Sustained payload (kg, log)')
    ax.set_title('Sustained two-arm capacity falls with hand distance, roughly like the NIOSH human limit',
                 fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='lower left', fontsize=8.8, facecolor=PANEL, edgecolor=GRID, labelcolor=INK, framealpha=0.95)
    fig.text(0.01, 0.005,
             'Pose: forearm horizontal, elbow drops toward the hip as the load comes in. Robots with a per-arm '
             'rating are sized at it; others at the median humanoid arm-actuator budget. Rated (continuous) torque.',
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


# ---------------------------------------------------------------------------
# 4) Holding power and runtime
# ---------------------------------------------------------------------------

def motor_km(flux: float = HEAT_FLUX_W_M2) -> pd.DataFrame:
    m = pd.read_csv(MOTOR_PATH).dropna(subset=['Cont_Torque_Nm', 'OD_mm', 'Length_mm', 'Weight_kg'])
    d, length = m['OD_mm'] / 1000, m['Length_mm'] / 1000
    area = np.pi * d * length + np.pi * d ** 2 / 2
    m = m.assign(P_cont_W=flux * area)
    return m.assign(Km=m['Cont_Torque_Nm'] / np.sqrt(m['P_cont_W']))


def holding_power(res: dict, payload_per_arm: float, h: float, km_fit, flux_scale: float = 1.0) -> float:
    """Copper loss (W) in both arms' pitch joints holding payload_per_arm at h."""
    b, A = km_fit
    t = res['table'].set_index('joint')
    factor = res['integration_factor']
    torques = joint_torques(res, payload_per_arm, h)
    watts = 0.0
    for group, joint in (('shoulder', 'Shoulder pitch'), ('elbow', 'Elbow pitch')):
        tau_m = torques[group] / t.loc[joint, 'ratio']        # holding: friction helps, no η
        km = A * (t.loc[joint, 'motor_kg'] / factor) ** b * np.sqrt(1 / flux_scale)
        watts += (tau_m / km) ** 2
    return 2 * watts


def plot_power(hum: pd.DataFrame, budget_frac: float, models, out: str) -> pd.DataFrame:
    km = motor_km()
    b, A, r2 = hs.power_fit(km['Weight_kg'].to_numpy(), km['Km'].to_numpy())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={'width_ratios': [1, 1.25]})
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        style(ax)

    ax1.scatter(km['Weight_kg'], km['Km'], s=22, color=BOUND_COLOR, alpha=0.6, edgecolors='none', zorder=2)
    xs = np.logspace(np.log10(km['Weight_kg'].min()), np.log10(km['Weight_kg'].max()), 50)
    for s, ls in ((1.0, '-'), *((s, ':') for s in HEAT_FLUX_BAND)):
        ax1.plot(xs, A * xs ** b / np.sqrt(s), color=INK if s == 1 else MUTED, linestyle=ls, linewidth=1.8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Motor mass (kg)')
    ax1.set_ylabel('Estimated Km (N·m / √W)')
    ax1.set_title(f'Km ∝ mass^{b:.2f} (R² {r2:.2f}, n={len(km)})', fontsize=12, fontweight='bold')
    ax1.text(0.03, 0.97, f'Km = τ_cont / √(q · surface area)\nq = {HEAT_FLUX_W_M2:.0f} W/m² assumed;\n'
             f'dotted: q × {HEAT_FLUX_BAND[0]:g} / × {HEAT_FLUX_BAND[1]:g}',
             transform=ax1.transAxes, va='top', fontsize=9, color=MUTED)

    by = hum.set_index('Name')
    rows = []
    for name, kwh in BATTERY_KWH.items():
        r = by.loc[name]
        res, basis = capacity_arm(r, budget_frac, models)
        per_arm = float(r['Arm_Payload_kg']) if pd.notna(r['Arm_Payload_kg']) else float(r['Payload_kg']) / 2
        h_full = res['lengths']['x_payload']
        # Hold the published load, or the sustained capacity at that distance if it is smaller
        close_load = min(per_arm, arm_capacity(res, CLOSE_M))
        full_load = min(per_arm, arm_capacity(res, h_full))
        p_avg = kwh * 1000 / float(r['Battery_h'])
        for label, load, h in ((f'{2 * close_load:.1f} kg at {CLOSE_M * 100:.0f} cm', close_load, CLOSE_M),
                               (f'{2 * full_load:.1f} kg at full reach', full_load, h_full)):
            p = [holding_power(res, load, h, (b, A), s) for s in (1.0, *HEAT_FLUX_BAND)]
            rows.append({'Name': name, 'Case': label, 'kWh': kwh, 'P_avg_W': p_avg, 'P_hold_W': p[0],
                         'Runtime_h': kwh * 1000 / (p_avg + p[0]),
                         'Runtime_lo_h': kwh * 1000 / (p_avg + max(p)),
                         'Runtime_hi_h': kwh * 1000 / (p_avg + min(p)),
                         'Published_h': float(r['Battery_h'])})
    df = pd.DataFrame(rows)

    y = 0
    ticks, labels = [], []
    for name, g in df.groupby('Name', sort=False):
        pub = g['Published_h'].iloc[0]
        ax2.barh(y, pub, color=MUTED, height=0.7, zorder=2)
        ax2.text(pub + 0.1, y, f'{pub:g} h published  (avg {g["P_avg_W"].iloc[0]:.0f} W)',
                 color=INK, fontsize=8.5, va='center')
        ticks.append(y)
        labels.append(f'{name}')
        for _, c in g.iterrows():
            y += 1
            ax2.barh(y, c['Runtime_h'], color=HUMANOID_COLOR, height=0.7, zorder=2)
            ax2.plot([c['Runtime_lo_h'], c['Runtime_hi_h']], [y, y], color='white', linewidth=1.2, zorder=3)
            ax2.text(max(c['Runtime_hi_h'], c['Runtime_h']) + 0.1, y,
                     f'{c["Runtime_h"]:.1f} h  (+{c["P_hold_W"]:.0f} W holding)', color=INK, fontsize=8.5,
                     va='center')
            ticks.append(y)
            labels.append(f'  holding {c["Case"]}')
        y += 1.6
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels, color=INK, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, df['Published_h'].max() * 1.45)
    ax2.set_xlabel('Runtime (h)')
    ax2.set_title('Runtime if both arms held the load the whole time', fontsize=12, fontweight='bold')
    fig.suptitle('Holding a load costs little battery close-in, and gets expensive at full reach',
                 fontsize=14, fontweight='bold', color=hs.TITLE, y=1.0)
    fig.text(0.01, -0.04,
             'Copper loss only, in the shoulder + elbow pitch joints of both arms; bare motor mass from '
             'the arm model. Holding torque = gravity torque ÷ ratio (gear friction ignored, it helps a static '
             'hold). Average power = battery kWh ÷ published runtime. White whisker: heat-flux band. '
             'Atlas, for comparison, publishes 4 h normal vs 2 h heavy.',
             color=MUTED, fontsize=8.5, wrap=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')
    return df


def main():
    hum = pd.read_csv(hs.HUMANOID_PATH)
    arms = pd.read_csv(hs.ARM_PATH).dropna(subset=['Payload_kg', 'Reach_m'])
    models = amodel.load_actuator_models()

    peak = plot_peak(peak_ratios(), OUT_PEAK)

    shares = hs.arm_actuator_shares(hum)['Share_full']
    budgets = tuple(float(q) for q in shares.quantile([0.25, 0.5, 0.75]))
    print(f'\nArm actuator budget (share of body mass) quartiles: '
          + ', '.join(f'{b * 100:.1f} %' for b in budgets) + f'  (n={len(shares)})')
    print(f'Joint-module peak factor: {peak:.2f}×')

    pl = plausibility(hum, budgets, peak, models)
    plot_plausibility(pl, budgets, peak, OUT_PLAUSIBLE)
    print('\nTwo-arm capacity (kg) at the median budget:')
    print(pl[['Name', 'Payload_kg', 'full_mid', 'close_mid', 'close_peak_mid', 'Verdict']]
          .round(1).to_string(index=False))

    plot_reach(hum, arms, budgets[1], models, OUT_REACH)

    pw = plot_power(hum, budgets[1], models, OUT_POWER)
    print('\nHolding power / runtime:')
    print(pw.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
