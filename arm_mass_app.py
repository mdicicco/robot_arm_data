"""
Interactive arm actuator-mass estimator.

Sizes wrist → elbow → shoulder actuators for a reach + payload using the
empirical link-length heuristic (analyze_arm_geometry) and the motor / gearbox
mass fits (arm_mass_model).
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analyze_arm_geometry as geo
import arm_mass_model as amodel

st.set_page_config(
    page_title="Arm Actuator Mass",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2d4a 50%, #0a1628 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121f36 0%, #0a1628 100%);
        border-right: 2px solid #2a4060;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a2d4a, #121f36);
        border: 1px solid #2a4060;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #e8f4fc !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #6b8ba4 !important;
    }
    hr { border-color: #2a4060; }
</style>
""",
    unsafe_allow_html=True,
)

INK = "#e8f4fc"
MUTED = "#6b8ba4"
GRID = "#2a4060"
PANEL = "#121f36"
LINK_COLOR = "#8aa4bd"
GROUP_COLORS = amodel.GROUP_COLORS
ARM_TYPE_COLORS = amodel.ARM_TYPE_COLORS

# Callout placement per joint: (dx, dy) label offset in px from the glyph.
# Pitch joints sit on the low row, roll joints on the high row so the tight wrist
# cluster never overlaps; everything below the arm is left for dimensions.
CALLOUT_OFFSETS = {
    "Shoulder yaw": (-150, 10),
    "Shoulder pitch": (-30, -110),
    "Upper-arm roll": (0, -210),
    "Elbow pitch": (0, -110),
    "Forearm roll": (-95, -210),
    "Wrist pitch": (0, -110),
    "Wrist roll": (95, -210),
}


@st.cache_resource
def models():
    return amodel.load_actuator_models()


@st.cache_data
def geometry_table():
    return geo.load_geometry()


@st.cache_data
def arm_catalog():
    return amodel.load_arm_catalog()


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,22,40,0.6)",
        font=dict(color=INK, family="JetBrains Mono, monospace", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID, font=dict(color=INK)),
    )
    layout.update(kw)
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Arm drawing
# ─────────────────────────────────────────────────────────────────────────────


def glyph_size(mass_kg: float, reach_m: float) -> float:
    """Actuator glyph radius (m): ∝ mass^(1/3), i.e. roughly its volume."""
    return float(np.clip(0.035 * reach_m * (mass_kg / 1.0) ** (1 / 3), 0.012 * reach_m, 0.09 * reach_m))


def draw_arm(res: dict) -> go.Figure:
    cfg = res["config"]
    L = res["lengths"]
    table = res["table"].set_index("joint")
    H = L["base"]
    R = cfg.reach_m
    fig = go.Figure()
    shapes = []

    # Floor + base pedestal
    x_base = -L["offset"]
    ped_w = 0.09 * R
    shapes.append(dict(type="line", x0=x_base - 0.35 * R, x1=L["x_payload"] + 0.25 * R, y0=0, y1=0,
                       line=dict(color=GRID, width=2)))
    shapes.append(dict(type="rect", x0=x_base - ped_w, x1=x_base + ped_w, y0=0, y1=0.05 * R,
                       fillcolor="#1a2d4a", line=dict(color=LINK_COLOR, width=1.5)))
    # Column: floor → shoulder yaw → (offset) → shoulder pitch
    col_w = 0.05 * R
    shapes.append(dict(type="rect", x0=x_base - col_w, x1=x_base + col_w, y0=0.05 * R, y1=H,
                       fillcolor="#1a2d4a", line=dict(color=LINK_COLOR, width=1.5)))
    if L["offset"] > 0:
        shapes.append(dict(type="line", x0=x_base, x1=0, y0=H, y1=H, line=dict(color=LINK_COLOR, width=12)))

    # Links (thickness shrinks distally)
    links = [
        ("upper", 0.0, L["x_elbow"], 14),
        ("forearm", L["x_elbow"], L["x_wrist"], 10),
        ("wrist", L["x_wrist"], L["x_flange"], 7),
    ]
    for _, a, b, w in links:
        shapes.append(dict(type="line", x0=a, x1=b, y0=H, y1=H, line=dict(color=LINK_COLOR, width=w)))
    if cfg.tool_offset_m > 0:
        shapes.append(dict(type="line", x0=L["x_flange"], x1=L["x_payload"], y0=H, y1=H,
                           line=dict(color=MUTED, width=3, dash="dot")))

    # Payload box mounted on the end of the arm, centered on the link axis
    pb = 0.035 * R * max(cfg.payload_kg, 0.2) ** (1 / 3)
    shapes.append(dict(type="rect", x0=L["x_payload"], x1=L["x_payload"] + pb,
                       y0=H - pb / 2, y1=H + pb / 2, fillcolor="#3a4a60", line=dict(color=INK, width=1.5)))

    # Actuator glyphs: pitch = circle (axis into page), roll/yaw = band along its axis
    hover_x, hover_y, hover_text, hover_color = [], [], [], []
    for j in res["joints"]:
        row = table.loc[j.name]
        color = GROUP_COLORS[j.group]
        r = glyph_size(row["actuator_kg"], R)
        if j.name == "Shoulder yaw":
            cx, cy = x_base, H - 2.2 * r
            shapes.append(dict(type="rect", x0=cx - r * 1.1, x1=cx + r * 1.1, y0=cy - r * 0.8, y1=cy + r * 0.8,
                               fillcolor=color, line=dict(color=INK, width=1.5 if j.sizing else 0.5)))
        elif "pitch" in j.name:
            cx, cy = j.x, H
            shapes.append(dict(type="circle", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                               fillcolor=color, line=dict(color=INK, width=2.5)))
        else:  # roll: band wrapped around the link
            cx, cy = j.x, H
            shapes.append(dict(type="rect", x0=cx - r * 0.8, x1=cx + r * 0.8, y0=cy - r * 0.75, y1=cy + r * 0.75,
                               fillcolor=color, line=dict(color=INK, width=0.5)))
        hover_x.append(cx)
        hover_y.append(cy)
        hover_color.append(color)
        hover_text.append(
            f"<b>{j.name}</b> ({j.group}{', sizing joint' if j.sizing else ''})<br>"
            f"Actuator {row['actuator_kg']:.3f} kg = gearbox {row['gearbox_kg']:.3f} + motor {row['motor_kg']:.3f}<br>"
            f"Joint torque {row['torque_nm']:.1f} Nm @ ratio {row['ratio']:.0f}<br>"
            f"Motor {row['motor_torque_nm']:.3f} Nm, {row['motor_speed_rpm']:.0f} rpm"
        )
        dx, dy = CALLOUT_OFFSETS.get(j.name, (0, -120))
        head = f"<b>{j.name}</b>" + (" ★" if j.sizing else "")
        body = f"{row['actuator_kg']:.2f} kg"
        if j.sizing:
            body += f"<br>{row['torque_nm']:.1f} Nm"
        fig.add_annotation(
            x=cx, y=cy, ax=dx, ay=dy, text=f"{head}<br>{body}",
            showarrow=True, arrowhead=0, arrowwidth=1.2, arrowcolor=color,
            font=dict(size=12, color=INK), align="center",
            bgcolor=PANEL, bordercolor=color, borderwidth=1.5, borderpad=5,
        )

    fig.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers",
        marker=dict(size=28, color="rgba(0,0,0,0)"),
        hovertext=hover_text, hoverinfo="text", showlegend=False,
    ))
    fig.add_annotation(
        x=L["x_payload"] + pb, y=H, ax=60, ay=55,
        text=f"<b>Payload</b><br>{cfg.payload_kg:.2f} kg", showarrow=True, arrowhead=0,
        arrowcolor=INK, font=dict(size=12, color=INK), bgcolor=PANEL, bordercolor=INK, borderwidth=1, borderpad=5,
    )

    # Dimension lines under the arm (lever arms), each labeled with % of reach
    y_dim = H - 0.13 * R
    dims = [
        ("upper", 0.0, L["x_elbow"]),
        ("forearm", L["x_elbow"], L["x_wrist"]),
        ("wrist", L["x_wrist"], L["x_flange"]),
    ]
    for name, a, b in dims:
        shapes.append(dict(type="line", x0=a, x1=b, y0=y_dim, y1=y_dim, line=dict(color=MUTED, width=1)))
        for xe in (a, b):
            shapes.append(dict(type="line", x0=xe, x1=xe, y0=y_dim - 0.015 * R, y1=y_dim + 0.015 * R,
                               line=dict(color=MUTED, width=1)))
        fig.add_annotation(x=(a + b) / 2, y=y_dim, yshift=-14, showarrow=False,
                           text=f"{name} {b - a:.3f} m ({(b - a) / R:.0%})", font=dict(size=11, color=MUTED))
    fig.add_annotation(x=x_base - col_w, y=H * 0.3, xshift=-8, xanchor="right", showarrow=False,
                       text=f"base<br>{H:.3f} m", font=dict(size=11, color=MUTED), align="right")

    x_lo = x_base - 0.45 * R
    x_hi = L["x_payload"] + 0.35 * R
    fig.update_xaxes(range=[x_lo, x_hi], visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-0.06 * R, H + 0.5 * R], visible=False)
    base_layout(fig, 560, shapes=shapes, margin=dict(l=0, r=0, t=10, b=0))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Supporting charts
# ─────────────────────────────────────────────────────────────────────────────


def mass_breakdown(res: dict) -> go.Figure:
    t = res["table"].iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=t["joint"], x=t["gearbox_kg"], orientation="h", name="Gearbox",
        marker=dict(color=[GROUP_COLORS[g] for g in t["group"]], line=dict(color="#0a1628", width=2)),
        hovertemplate="%{y}<br>Gearbox %{x:.3f} kg<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=t["joint"], x=t["motor_kg"], orientation="h", name="Motor",
        marker=dict(color=[GROUP_COLORS[g] for g in t["group"]], opacity=0.5,
                    pattern=dict(shape="/", fgcolor=INK, size=6), line=dict(color="#0a1628", width=2)),
        text=[f"{v:.2f} kg" for v in t["actuator_kg"]], textposition="outside",
        textfont=dict(color=INK), hovertemplate="%{y}<br>Motor %{x:.3f} kg<extra></extra>",
    ))
    base_layout(fig, 330, barmode="stack", showlegend=False, title=dict(text="Actuator mass: gearbox (solid) + motor (hatched)", font=dict(size=13)),
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK)), margin=dict(l=10, r=10, t=70, b=10))
    fig.update_xaxes(title="kg", gridcolor=GRID, zeroline=False, color=MUTED,
                     range=[0, float(t["actuator_kg"].max()) * 1.25])
    fig.update_yaxes(color=INK)
    return fig


def convergence(res: dict) -> go.Figure:
    h = res["history"]
    fig = go.Figure()
    for g in amodel.GROUPS:
        fig.add_trace(go.Scatter(
            x=h["iteration"], y=h[g], mode="lines+markers", name=g,
            line=dict(color=GROUP_COLORS[g], width=2), marker=dict(size=8),
            hovertemplate=f"{g}<br>iter %{{x}}: %{{y:.4f}} kg<extra></extra>",
        ))
    base_layout(fig, 300, title=dict(text=f"Fixed-point iteration ({res['iterations']} passes)", font=dict(size=13)),
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK)), margin=dict(l=10, r=10, t=70, b=10), hovermode="x unified")
    fig.update_xaxes(title="Iteration (wrist → elbow → shoulder each pass)", gridcolor=GRID, dtick=1, color=MUTED)
    fig.update_yaxes(title="Actuator kg", gridcolor=GRID, color=MUTED)
    return fig


@st.cache_data
def payload_ratio_sweep(key: str, _cfg: amodel.ArmConfig) -> pd.DataFrame:
    """Sweep payloads × reaches with the current settings (cached on `key`)."""
    return amodel.payload_ratio_sweep(_cfg, models=models())


PLOTLY_SYMBOLS = {"collaborative": "circle", "industrial": "square", "research": "diamond", "hobby": "triangle-up"}


def payload_ratio_chart(res: dict, sweep: pd.DataFrame, highlight: float | None = None) -> go.Figure:
    cfg = res["config"]
    arms = arm_catalog().copy()
    arms["bin"] = amodel.assign_payload_bin(arms["Payload_kg"])
    edges = amodel.payload_bin_edges()
    symbols = arms["Type"].map(PLOTLY_SYMBOLS).fillna("circle")
    hover = ("%{customdata[0]} %{customdata[1]} (%{customdata[4]})<br>%{customdata[2]} kg payload / "
             "%{customdata[3]} kg arm<br>reach %{x:.2f} m · ratio %{y:.3f}<extra></extra>")

    def custom(d):
        return np.stack([d["MFG"], d["Name"], d["Payload_kg"], d["Weight_kg"], d["Type"]], axis=-1)

    def emphasis(payload):
        """(dot opacity, dot size, line opacity, line width, dot color override)"""
        if highlight is None:
            return 0.9, 9, 1.0, 2.5, None
        if payload == highlight:
            return 1.0, 12, 1.0, 4.0, None
        return 0.18, 6, 0.2, 1.5, amodel.OUT_OF_BAND_COLOR

    fig = go.Figure()
    far = arms[arms["bin"].isna()]
    fig.add_trace(go.Scatter(
        x=far["Reach_m"], y=far["Payload_Ratio"], mode="markers",
        name=f"outside {edges[0]:.2g}–{edges[-1]:.2g} kg ({len(far)})", legendgroup="far", legendrank=200,
        marker=dict(size=6, color=amodel.OUT_OF_BAND_COLOR, opacity=0.35 if highlight is None else 0.12,
                    symbol=symbols[far.index]),
        customdata=custom(far), hovertemplate=hover,
    ))
    for i, (payload, color) in enumerate(zip(amodel.SWEEP_PAYLOADS, amodel.SWEEP_COLORS)):
        near = arms[arms["bin"] == payload]
        group = f"p{payload:g}"
        label = f"{payload:g} kg · arms {edges[i]:.2g}–{edges[i + 1]:.2g} kg ({len(near)})"
        dot_op, dot_size, line_op, line_w, override = emphasis(payload)
        focus = highlight is not None and payload == highlight
        d = sweep[sweep["payload_kg"] == payload]
        dots = go.Scatter(
            x=near["Reach_m"], y=near["Payload_Ratio"], mode="markers", name=label, legendgroup=group,
            showlegend=False,
            marker=dict(size=dot_size, color=override or color, opacity=dot_op, symbol=symbols[near.index],
                        line=dict(color=INK if focus else "#0a1628", width=1)),
            customdata=custom(near), hovertemplate=hover,
        )
        halo = go.Scatter(  # surface halo so lines read over dots
            x=d["reach_m"], y=d["payload_ratio"], mode="lines", legendgroup=group, showlegend=False,
            line=dict(color="#0a1628", width=line_w + 3.5), opacity=line_op, hoverinfo="skip",
        )
        line = go.Scatter(
            x=d["reach_m"], y=d["payload_ratio"], mode="lines", name=label, legendgroup=group, legendrank=100 + i,
            line=dict(color=color, width=line_w), opacity=line_op, customdata=d["actuator_kg"],
            hovertemplate=f"{payload:g} kg bound<br>reach %{{x:.2f}} m<br>actuators %{{customdata:.2f}} kg"
                          "<br>ratio %{y:.3f}<extra></extra>",
        )
        if focus:
            focused = (dots, halo, line)  # drawn last so it sits on top
        else:
            fig.add_traces([dots, halo, line])
        end = d.iloc[-1]
        fig.add_annotation(x=end["reach_m"], y=np.log10(end["payload_ratio"]), text=f"{payload:g} kg",
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=13 if focus else 11, color=INK if line_op == 1.0 else MUTED))
    if highlight is not None:
        fig.add_traces(list(focused))
    for typ, sym in PLOTLY_SYMBOLS.items():  # shape key only
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=typ, legendgroup="shape", legendrank=300,
            legendgrouptitle_text="Marker = arm type",
            marker=dict(size=9, color="#a9b8c8", symbol=sym),
        ))
    arm_kg = res["actuator_total_kg"] + res["structure_kg"]
    fig.add_trace(go.Scatter(
        x=[cfg.reach_m], y=[cfg.payload_kg / arm_kg], mode="markers", name="this design", legendrank=400,
        marker=dict(size=18, symbol="star", color=INK, line=dict(color="#0a1628", width=2)),
        hovertemplate=f"This design: {cfg.payload_kg:.1f} kg @ %{{x:.2f}} m<br>ratio %{{y:.3f}}<extra></extra>",
    ))
    y_top = max(2.0, float(sweep["payload_ratio"].max()), cfg.payload_kg / arm_kg) * 1.4
    base_layout(fig, 560, title=dict(text="Payload ratio vs reach — bound curves vs arms colored by nearest curve payload "
                                          "(click a legend entry to isolate a band)", font=dict(size=13)),
                legend=dict(bgcolor="rgba(18,31,54,0.85)", bordercolor=GRID, borderwidth=1, font=dict(color=INK, size=11),
                            grouptitlefont=dict(color=MUTED, size=11)),
                margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(title="Reach (m)", range=[0, 4.8], gridcolor=GRID, zeroline=False, color=MUTED)
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    fig.update_yaxes(title="Payload ratio (payload / arm mass)", type="log", range=[-2, np.log10(y_top)],
                     tickvals=ticks, ticktext=[f"{v:g}" for v in ticks], gridcolor=GRID, color=MUTED)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────


def main():
    mdl = models()
    geo_df = geometry_table()

    with st.sidebar:
        st.markdown("# Arm Config")
        reach = st.slider("Reach (m)", 0.20, 2.50, 0.85, 0.01, format="%.2f")
        payload = st.slider("Payload (kg)", 0.1, 50.0, 5.0, 0.1, format="%.1f")
        dof = st.segmented_control("DOF", [6, 7], default=6, format_func=lambda d: f"{d}-DOF") or 6

        st.markdown("### Actuators")
        gearbox = st.segmented_control(
            "Gearbox type", amodel.GEARBOX_TYPES, default=amodel.ALL,
            help="'all' = pooled fit over every gearbox type (the average design).",
        ) or amodel.ALL
        motor = st.segmented_control(
            "Motor type", amodel.MOTOR_FORMS, default=amodel.ALL,
            help="'all' = pooled fit over every motor form (the average design).",
        ) or amodel.ALL

        st.markdown("### Geometry")
        basis = st.segmented_control(
            "Heuristic from", list(geo.GEOMETRY_BASES), default="all",
            help="Which arms from data/arm_geometry_data.csv the link-length fractions come from.",
        ) or "all"
        fitted = amodel.default_geometry(basis)

        with st.expander("Link fractions (override)"):
            st.caption(f"Medians from {fitted.n} arms; renormalized to sum to 1.")
            f_off = st.slider("Shoulder offset", 0.0, 0.20, round(fitted.offset, 3), 0.005, key=f"off{basis}")
            f_up = st.slider("Upper arm", 0.20, 0.60, round(fitted.upper, 3), 0.005, key=f"up{basis}")
            f_fa = st.slider("Forearm", 0.20, 0.60, round(fitted.forearm, 3), 0.005, key=f"fa{basis}")
            f_wr = st.slider("Wrist → flange", 0.03, 0.30, round(fitted.wrist, 3), 0.005, key=f"wr{basis}")
            f_base = st.slider("Base height", 0.05, 0.60, round(fitted.base, 3), 0.005, key=f"ba{basis}")
        geometry = geo.GeometryFractions(f_base, f_off, f_up, f_fa, f_wr, n=fitted.n).normalized()

        with st.expander("Sizing assumptions"):
            sf = st.slider("Safety factor on static torque", 1.0, 4.0, 1.0, 0.1,
                           help="Covers acceleration / dynamics on top of the horizontal gravity case.")
            tool = st.slider("Payload CoG beyond flange (m)", 0.0, 0.30, 0.0, 0.005)
            integrated = st.toggle(
                "Apply joint-module integration factor", value=True,
                help="Scale bare gearbox+motor by median(real module mass / predicted) from robot_joint_data.csv.",
            )
            st.caption(
                "Integration factors: "
                + ", ".join(f"{t} {f:.2f}" for t, f in mdl.integration.items())
            )

        default_ratio = amodel.DEFAULT_RATIO[gearbox]
        with st.expander("Gear ratio & joint speed per cluster"):
            ratio, speed = {}, {}
            defaults_dps = amodel.ArmConfig(1, 1).joint_speed_dps
            for g in ("shoulder", "elbow", "wrist"):
                ratio[g] = st.slider(f"{g} ratio", 3.0, 160.0, default_ratio, 1.0, key=f"r_{g}_{gearbox}")
                speed[g] = st.slider(f"{g} speed (deg/s)", 30.0, 720.0, defaults_dps[g], 10.0, key=f"s_{g}")

        with st.expander("Structure (placeholder)"):
            link_mpm = st.slider("Link mass (kg per m)", 0.0, 20.0, 0.0, 0.1,
                                 help="Uniform-rod stand-in until the structure model exists. It's carried in the same iteration loop.")

    cfg = amodel.ArmConfig(
        reach_m=reach, payload_kg=payload, dof=int(dof), gearbox_type=gearbox, motor_form=motor,
        geometry=geometry, ratio=ratio, joint_speed_dps=speed, safety_factor=sf, tool_offset_m=tool,
        integrated=integrated, link_mass_per_m=link_mpm,
    )
    res = amodel.size_arm(cfg, mdl)
    grp = res["groups"]
    t = res["table"]

    st.markdown("# Arm Actuator Mass Estimator")
    st.caption(
        f"{dof}-DOF · {gearbox} gearbox + {motor} motor · sized distal → proximal at full horizontal extension · "
        f"SF {sf:.1f} · converged in {res['iterations']} iterations"
    )

    c = st.columns(5)
    c[0].metric("Total actuators", f"{res['actuator_total_kg']:.2f} kg",
                help=f"{len(t)} actuators" + (f" + {res['structure_kg']:.2f} kg structure placeholder" if res["structure_kg"] else ""))
    for col, g in zip(c[1:4], ("shoulder", "elbow", "wrist")):
        n = int((t["group"] == g).sum())
        col.metric(f"{g.capitalize()} ×{n} · {grp[g]['torque_nm']:.1f} Nm", f"{grp[g]['actuator_kg']:.2f} kg",
                   help=f"Each {g} actuator; sized by {grp[g]['sizing_joint'].lower()} torque")
    c[4].metric("Payload / actuator mass", f"{payload / res['actuator_total_kg']:.2f}")

    st.plotly_chart(draw_arm(res), use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "★ = sizing joint for its cluster; all joints in a cluster share its actuator. "
        "Circles = pitch axes (into page), bands = roll/yaw axes. Glyph size ∝ mass^⅓. Hover for motor/gearbox detail."
    )

    for w in res["warnings"]:
        st.warning(w, icon="⚠️")

    # Curves always use the pooled "all" gearbox + motor; the buttons only move the star.
    template = replace(cfg, reach_m=1.0, payload_kg=1.0, gearbox_type=amodel.ALL, motor_form=amodel.ALL, ratio={})
    sweep = payload_ratio_sweep(repr(template), template)
    band_labels = {"all": None, **{f"{p:g} kg": p for p in amodel.SWEEP_PAYLOADS}}
    band = st.segmented_control(
        "Highlight payload band", list(band_labels), default="all",
        help="Emphasize one bound curve and the published arms whose payload is nearest to it.",
    ) or "all"
    highlight = band_labels[band]
    if highlight is not None:
        edges = amodel.payload_bin_edges()
        i = amodel.SWEEP_PAYLOADS.index(highlight)
        arms = arm_catalog()
        near = arms[amodel.assign_payload_bin(arms["Payload_kg"]) == highlight]
        below = int((near["Payload_Ratio"] <= np.interp(near["Reach_m"], sweep.loc[sweep.payload_kg == highlight, "reach_m"],
                                                          sweep.loc[sweep.payload_kg == highlight, "payload_ratio"])).sum())
        st.caption(
            f"{len(near)} published arms with payload {edges[i]:.2g}–{edges[i + 1]:.2g} kg · "
            f"{below} below the {highlight:g} kg bound, {len(near) - below} above "
            f"(reach outside {amodel.SWEEP_REACHES[0]:g}–{amodel.SWEEP_REACHES[-1]:g} m compared at the nearest end)"
        )
    st.plotly_chart(payload_ratio_chart(res, sweep, highlight), use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"Curves sweep reach {amodel.SWEEP_REACHES[0]:g}–{amodel.SWEEP_REACHES[-1]:g} m for each payload with the "
        f"pooled 'all' gearbox + motor fits (ratio {amodel.DEFAULT_RATIO[amodel.ALL]:g}, η {amodel.GEAR_EFFICIENCY[amodel.ALL]:.2f}) "
        "— an average actuator design. DOF, geometry, SF, speeds and the structure placeholder follow the sidebar. "
        "The gearbox / motor buttons only move the ★. Arm mass = actuators only, so a real arm sits below its curve. "
        "Dots take the color of the curve nearest their payload (log-midpoint bands); gray = payload outside every band. "
        "Click a legend entry to hide a band, double-click to isolate it."
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(mass_breakdown(res), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.plotly_chart(convergence(res), use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Sizing chain")
    chain = pd.DataFrame(
        [
            {
                "cluster": g,
                "sizing joint": grp[g]["sizing_joint"],
                "payload lever (m)": grp[g]["lever_m"],
                "static (Nm)": grp[g]["static_torque_nm"],
                "sized (Nm)": grp[g]["torque_nm"],
                "ratio": grp[g]["ratio"],
                "motor (Nm)": grp[g]["motor_torque_nm"],
                "motor (rpm)": grp[g]["motor_speed_rpm"],
                "actuator (kg)": grp[g]["actuator_kg"],
            }
            for g in amodel.GROUPS
        ]
    )
    st.dataframe(chain.style.format(precision=3), hide_index=True, use_container_width=True)
    L = res["lengths"]
    st.markdown(
        f"Links: base **{L['base']:.3f}** · offset **{L['offset']:.3f}** · upper **{L['upper']:.3f}** · "
        f"forearm **{L['forearm']:.3f}** · wrist **{L['wrist']:.3f}** m"
    )

    with st.expander("Per-joint table"):
        st.dataframe(t.style.format(precision=3), hide_index=True, use_container_width=True)
    with st.expander(f"Geometry source arms ({len(geo_df)})"):
        cols = ["Name", "MFG", "Type", "DOF", "Reach_m", "frac_base", "frac_offset", "frac_upper",
                "frac_forearm", "frac_wrist", "Confidence"]
        st.dataframe(geo_df[cols].style.format(precision=3), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
