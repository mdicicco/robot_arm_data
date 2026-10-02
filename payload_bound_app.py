"""
Live payload-bound explorer.

Scatter of every published arm (reach vs payload factor) with:

  1. Theoretical actuator-only ceiling (from size_arm sweep):
         PF_max(R, P) = 0.690 · P^0.290 · R^(-0.848)

  2. Empirical single-parameter warps of that curve:
         PF_fit(R, P) = η · PF_max(R, P)
     η fitted per Type, plus one pooled "all" η (log-space mean ratio).

Slider sets payload P; arms within ±tol of that payload are highlighted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Closed-form fit to the pooled "all" size_arm sweep (actuator-only upper bound).
THEORY_A = 0.690
THEORY_ALPHA = 0.290
THEORY_BETA = -0.848

TYPE_COLORS = {
    "collaborative": "#2ecc71",
    "industrial": "#e74c3c",
    "research": "#9b59b6",
    "hobby": "#3498db",
}
ALL_COLOR = "#f1c40f"
THEORY_COLOR = "#00d4ff"

st.set_page_config(
    page_title="Payload Bound Explorer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background: linear-gradient(135deg, #0a1628 0%, #1a2d4a 50%, #0a1628 100%); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121f36 0%, #0a1628 100%);
        border-right: 2px solid #2a4060;
    }
    h1, h2, h3 { color: #00d4ff !important; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a2d4a, #121f36);
        border: 1px solid #2a4060; border-radius: 12px; padding: 12px;
    }
    div[data-testid="stMetricValue"] { color: #7bed9f !important; }
    div[data-testid="stMetricLabel"] { color: #6b8ba4 !important; }
</style>
""",
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class ScaleFit:
    """Single-parameter warp: PF ≈ η · PF_theory(R, P)."""

    label: str
    n: int
    eta: float
    mape: float

    def pf(self, reach_m: np.ndarray | float, payload_kg: float) -> np.ndarray:
        return self.eta * pf_theory(reach_m, payload_kg)

    def formula(self) -> str:
        return (
            f"{self.eta:.3g} · {THEORY_A:g} · P^{THEORY_ALPHA:g} · R^{THEORY_BETA:g}"
        )


@st.cache_data
def load_arms() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "data", "robot_arm_data.csv")
    df = pd.read_csv(path).dropna(subset=["Payload_kg", "Reach_m", "Weight_kg"]).copy()
    df = df[(df["Payload_kg"] > 0) & (df["Reach_m"] > 0) & (df["Weight_kg"] > 0)]
    df["Payload_Factor"] = df["Payload_kg"] / df["Weight_kg"]
    return df.reset_index(drop=True)


def pf_theory(reach_m: np.ndarray | float, payload_kg: float) -> np.ndarray:
    return THEORY_A * (payload_kg ** THEORY_ALPHA) * (np.asarray(reach_m, dtype=float) ** THEORY_BETA)


def fit_scale(df: pd.DataFrame, label: str) -> ScaleFit | None:
    """η = exp(mean[log(PF_data) − log(PF_theory)]) — one free parameter."""
    d = df.dropna(subset=["Payload_kg", "Reach_m", "Payload_Factor"]).copy()
    d = d[(d["Payload_kg"] > 0) & (d["Reach_m"] > 0) & (d["Payload_Factor"] > 0)]
    if len(d) < 5:
        return None
    theor = pf_theory(d["Reach_m"].to_numpy(), d["Payload_kg"].to_numpy())
    # Element-wise theory at each arm's own (R, P)
    theor = THEORY_A * (d["Payload_kg"].to_numpy() ** THEORY_ALPHA) * (
        d["Reach_m"].to_numpy() ** THEORY_BETA
    )
    eta = float(np.exp(np.mean(np.log(d["Payload_Factor"].to_numpy()) - np.log(theor))))
    pred = eta * theor
    mape = float(np.mean(np.abs(pred - d["Payload_Factor"]) / d["Payload_Factor"]) * 100.0)
    return ScaleFit(label=label, n=len(d), eta=eta, mape=mape)


@st.cache_data
def fit_all_groups(arms: pd.DataFrame) -> dict[str, ScaleFit]:
    fits: dict[str, ScaleFit] = {}
    for typ in ("collaborative", "industrial", "research", "hobby"):
        f = fit_scale(arms[arms["Type"] == typ], typ)
        if f is not None:
            fits[typ] = f
    f_all = fit_scale(arms, "all")
    if f_all is not None:
        fits["all"] = f_all
    return fits


def main() -> None:
    arms = load_arms()
    fits = fit_all_groups(arms)

    st.title("Payload-factor bound vs single-η warps")
    st.caption(
        f"Theory: PF_max = {THEORY_A:g} · P^{THEORY_ALPHA:g} · R^{THEORY_BETA:g}.  "
        "Empirical curves use one scale η per group: PF ≈ η · PF_max (shape unchanged)."
    )

    with st.sidebar:
        st.header("Payload")
        p_min = float(max(0.25, arms["Payload_kg"].min()))
        p_max = float(min(50.0, arms["Payload_kg"].max()))
        payload = st.slider("Payload mass (kg)", p_min, p_max, 5.0, 0.25)
        tol_pct = st.slider("Highlight band (±%)", 1, 50, 10, 1)
        st.markdown("---")
        show_types = st.multiselect(
            "Types",
            options=sorted(arms["Type"].dropna().unique()),
            default=sorted(arms["Type"].dropna().unique()),
        )
        st.markdown("---")
        st.header("Curve blend")
        blend_options = [k for k in ("all", "collaborative", "industrial", "research", "hobby") if k in fits]
        blend_base = st.selectbox(
            "0% = data fit from",
            options=blend_options,
            index=0,
            help="η at 0% comes from this group's single-parameter warp; 100% is theory (η=1).",
        )
        blend_pct = st.slider(
            "Blend toward theory",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            format="%d%%",
            help="0% = data-fit η · PF_max ·  100% = theoretical PF_max (η=1).",
        )
        eta0 = fits[blend_base].eta
        t = blend_pct / 100.0
        eta_blend = eta0 + t * (1.0 - eta0)
        st.caption(f"η = {eta_blend:.3g}  (data η={eta0:.3g} → theory η=1)")
        st.markdown("---")
        st.header("Overlays")
        show_theory = st.checkbox("Theoretical PF_max (η=1)", value=True)
        show_all_fit = st.checkbox("Warp: all", value=False)
        show_type_fits = st.checkbox("Warp: per type", value=False)
        log_y = st.checkbox("Log payload-factor axis", value=False)

    visible = arms[arms["Type"].isin(show_types)].copy()
    lo, hi = payload * (1 - tol_pct / 100.0), payload * (1 + tol_pct / 100.0)
    near = visible[(visible["Payload_kg"] >= lo) & (visible["Payload_kg"] <= hi)]
    far = visible.drop(near.index)

    reach_line = np.linspace(0.2, 4.0, 200)
    fig = go.Figure()

    for typ, color in TYPE_COLORS.items():
        d = far[far["Type"] == typ]
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["Reach_m"],
                y=d["Payload_Factor"],
                mode="markers",
                name=f"{typ} (out of band)",
                legendgroup=typ,
                marker=dict(size=7, color=color, opacity=0.18, line=dict(width=0)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "P=%{customdata[2]:.2g} kg  W=%{customdata[3]:.2g} kg<br>"
                    "R=%{x:.2f} m  PF=%{y:.3f}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [d["Name"], d["MFG"], d["Payload_kg"], d["Weight_kg"]]
                ),
                showlegend=False,
            )
        )

    for typ, color in TYPE_COLORS.items():
        d = near[near["Type"] == typ]
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["Reach_m"],
                y=d["Payload_Factor"],
                mode="markers",
                name=f"{typ} (±{tol_pct:g}%)",
                legendgroup=typ,
                marker=dict(
                    size=11,
                    color=color,
                    opacity=0.95,
                    line=dict(width=1.2, color="#e8f4fc"),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "P=%{customdata[2]:.2g} kg  W=%{customdata[3]:.2g} kg<br>"
                    "R=%{x:.2f} m  PF=%{y:.3f}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [d["Name"], d["MFG"], d["Payload_kg"], d["Weight_kg"]]
                ),
            )
        )

    if show_theory:
        fig.add_trace(
            go.Scatter(
                x=reach_line,
                y=pf_theory(reach_line, payload),
                mode="lines",
                name=f"Theory PF_max @ {payload:g} kg",
                line=dict(color=THEORY_COLOR, width=2, dash="dash"),
                hovertemplate="R=%{x:.2f} m<br>PF_max=%{y:.3f}<extra></extra>",
            )
        )

    if show_all_fit and "all" in fits:
        f = fits["all"]
        fig.add_trace(
            go.Scatter(
                x=reach_line,
                y=f.pf(reach_line, payload),
                mode="lines",
                name=f"η_all={f.eta:.3g}",
                line=dict(color=ALL_COLOR, width=2, dash="dot"),
                hovertemplate="R=%{x:.2f} m<br>PF_fit=%{y:.3f}<extra></extra>",
            )
        )

    if show_type_fits:
        for typ, color in TYPE_COLORS.items():
            if typ not in fits or typ not in show_types:
                continue
            f = fits[typ]
            fig.add_trace(
                go.Scatter(
                    x=reach_line,
                    y=f.pf(reach_line, payload),
                    mode="lines",
                    name=f"η_{typ}={f.eta:.3g}",
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate="R=%{x:.2f} m<br>PF_fit=%{y:.3f}<extra></extra>",
                )
            )

    # Primary blended curve: η(t) = η0 + t·(1−η0)
    blend_y = eta_blend * pf_theory(reach_line, payload)
    fig.add_trace(
        go.Scatter(
            x=reach_line,
            y=blend_y,
            mode="lines",
            name=f"Blend {blend_pct:g}%  (η={eta_blend:.3g})",
            line=dict(color="#e8f4fc", width=3.5),
            hovertemplate="R=%{x:.2f} m<br>PF_blend=%{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0a1628",
        height=620,
        margin=dict(l=60, r=20, t=30, b=50),
        legend=dict(
            bgcolor="rgba(18,31,54,0.9)",
            bordercolor="#2a4060",
            borderwidth=1,
            font=dict(color="#e8f4fc"),
        ),
        xaxis=dict(
            title="Reach (m)",
            gridcolor="#2a4060",
            zeroline=False,
            range=[0.0, 4.0],
        ),
        yaxis=dict(
            title="Payload Factor (Payload / Robot Mass)",
            gridcolor="#2a4060",
            zeroline=False,
            type="log" if log_y else "linear",
            range=[np.log10(0.05), np.log10(2.0)] if log_y else [0.0, 2.0],
        ),
        font=dict(color="#e8f4fc"),
        hovermode="closest",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Payload P", f"{payload:g} kg")
    c2.metric("Band", f"{lo:.2g} – {hi:.2g} kg")
    c3.metric("Highlighted", f"{len(near)} / {len(visible)}")
    c4.metric(f"η blend ({blend_pct:g}%)", f"{eta_blend:.3g}")

    # White-line equation with live η and P
    a_eff = eta_blend * THEORY_A
    st.markdown(
        f"**White curve**  ·  "
        f"$\\mathrm{{PF}}(R) = {eta_blend:.4g}\\,\\cdot\\,{THEORY_A:g}"
        f"\\,\\cdot\\,{payload:g}^{{{THEORY_ALPHA:g}}}"
        f"\\,\\cdot\\,R^{{{THEORY_BETA:g}}}$"
        f"  $=\\;{a_eff:.4g}\\,\\cdot\\,R^{{{THEORY_BETA:g}}}$"
        f"  &nbsp;&nbsp;*(P = {payload:g} kg fixed; η from {blend_base} @ {blend_pct:g}%)*"
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.subheader("Single-parameter warps  ·  PF ≈ η · PF_max(R, P)")
    rows = [
        {
            "curve": "theory",
            "n": "—",
            "η": 1.0,
            "MAPE": "—",
            "formula": f"{THEORY_A:g} · P^{THEORY_ALPHA:g} · R^{THEORY_BETA:g}",
        }
    ]
    for key in ("all", "collaborative", "industrial", "research", "hobby"):
        if key not in fits:
            continue
        f = fits[key]
        rows.append(
            {
                "curve": key,
                "n": f.n,
                "η": f.eta,
                "MAPE": f.mape / 100.0,
                "formula": f.formula(),
            }
        )
    coef = pd.DataFrame(rows)
    st.dataframe(
        coef.style.format(
            {
                "η": "{:.3g}",
                "MAPE": lambda v: "—" if v == "—" else f"{v:.0%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader(f"Arms within ±{tol_pct:g}% of {payload:g} kg")
    if near.empty:
        st.info("No published arms in this payload band.")
    else:
        table = near[
            ["Name", "MFG", "Type", "Payload_kg", "Reach_m", "Weight_kg", "Payload_Factor"]
        ].copy()
        table["PF_theory"] = pf_theory(table["Reach_m"].to_numpy(), payload)
        table["PF_blend"] = eta_blend * table["PF_theory"].to_numpy()
        table["PF / blend"] = table["Payload_Factor"] / table["PF_blend"]
        table = table.sort_values("Payload_Factor", ascending=False)
        st.dataframe(
            table.style.format(
                {
                    "Payload_kg": "{:.2g}",
                    "Reach_m": "{:.3f}",
                    "Weight_kg": "{:.2g}",
                    "Payload_Factor": "{:.3f}",
                    "PF_theory": "{:.3f}",
                    "PF_blend": "{:.3f}",
                    "PF / blend": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
