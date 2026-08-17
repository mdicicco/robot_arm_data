"""
Interactive robot joint price estimator.
Fits a log-log Ridge model on priced modules in data/robot_joint_data.csv.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from joint_price_model import (
    TYPE_ACCURACY_ARCMIN,
    TYPE_ENCODER_BITS,
    estimate_price,
    fit_models,
    load_joints,
    metrics,
    prepare_features,
)

st.set_page_config(
    page_title="Robot Joint Price Estimator",
    page_icon="⚙️",
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
        color: #7bed9f !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #6b8ba4 !important;
    }
    .stSlider label, .stSelectbox label, .stToggle label {
        color: #e8f4fc !important;
    }
    hr { border-color: #2a4060; }
</style>
""",
    unsafe_allow_html=True,
)

TYPE_COLORS = {
    "qdd": "#2ecc71",
    "planetary": "#1abc9c",
    "harmonic": "#3498db",
    "cycloidal": "#9b59b6",
    "series-elastic": "#e74c3c",
    "hobby-servo": "#f39c12",
}


@st.cache_data
def load_raw():
    return load_joints()


@st.cache_resource
def load_fitted():
    return fit_models(load_raw())


def main():
    raw = load_raw()
    prepared = prepare_features(raw)
    fitted = load_fitted()
    stats = metrics(fitted)
    types = sorted(prepared["Type"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown("# Joint Config")
        st.markdown("---")
        joint_type = st.selectbox("Joint type", types, index=types.index("qdd") if "qdd" in types else 0)
        st.markdown("### Specs")
        torque = st.slider("Peak torque (Nm)", 0.5, 400.0, 20.0, 0.5)
        speed = st.slider("Rated speed (rpm)", 5.0, 600.0, 120.0, 1.0)
        od = st.slider("Outer diameter (mm)", 25.0, 180.0, 90.0, 1.0)
        accuracy = st.slider(
            "Accuracy (arcmin)",
            0.1,
            30.0,
            float(TYPE_ACCURACY_ARCMIN.get(joint_type, 12.0)),
            0.1,
            help="Lower is more precise. Defaults follow typical backlash/repeatability for the class.",
        )
        encoder_bits = st.slider(
            "Encoder bits",
            10,
            22,
            int(TYPE_ENCODER_BITS.get(joint_type, 14)),
            1,
        )
        st.markdown("### Features")
        dual_encoder = st.toggle("Dual encoder", value=joint_type in {"qdd", "harmonic", "series-elastic"})
        has_brake = st.toggle("Brake", value=False)
        has_driver = st.toggle("Integrated driver", value=True)
        st.markdown("### Mass")
        auto_mass = st.toggle("Auto-estimate mass", value=True)
        if auto_mass:
            preview = estimate_price(
                fitted,
                torque_nm=torque,
                speed_rpm=speed,
                od_mm=od,
                mass_kg=None,
                accuracy_arcmin=accuracy,
                joint_type=joint_type,
                encoder_bits=float(encoder_bits),
                dual_encoder=dual_encoder,
                has_brake=has_brake,
                has_driver=has_driver,
            )
            mass = preview["mass_kg"]
            st.info(f"Estimated: **{mass:.2f} kg**")
        else:
            mass = st.slider("Mass (kg)", 0.05, 12.0, 0.60, 0.01)

    result = estimate_price(
        fitted,
        torque_nm=torque,
        speed_rpm=speed,
        od_mm=od,
        mass_kg=None if auto_mass else mass,
        accuracy_arcmin=accuracy,
        joint_type=joint_type,
        encoder_bits=float(encoder_bits),
        dual_encoder=dual_encoder,
        has_brake=has_brake,
        has_driver=has_driver,
    )
    mass = result["mass_kg"]
    cost = result["cost_usd"]
    usd_per_nm = cost / torque if torque else 0.0
    torque_density = torque / mass if mass else 0.0

    st.markdown("# Robot Joint Price Estimator")
    st.markdown("*Two-stage fit: type + torque, then residual adjustments for speed, size, mass, accuracy, features*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated price", f"${cost:,.0f}")
    c2.metric("Price / Nm", f"${usd_per_nm:,.1f}")
    c3.metric("Mass", f"{mass:.2f} kg", "estimated" if auto_mass else "manual")
    c4.metric("Torque density", f"{torque_density:.1f} Nm/kg")

    st.markdown("---")

    priced = fitted.train.copy()
    type_data = priced[priced["Type"] == joint_type]
    fig = go.Figure()
    others = priced[priced["Type"] != joint_type]
    if len(others):
        fig.add_trace(
            go.Scatter(
                x=others["Torque_Nm"],
                y=others["Cost_USD"],
                mode="markers",
                name="Other types",
                marker=dict(size=8, color="#4a6080", opacity=0.45),
                text=others.apply(lambda r: f"{r['Name']} ({r['MFG']})", axis=1),
                hovertemplate="%{text}<br>Torque: %{x:.1f} Nm<br>Price: $%{y:.0f}<extra></extra>",
            )
        )
    if len(type_data):
        fig.add_trace(
            go.Scatter(
                x=type_data["Torque_Nm"],
                y=type_data["Cost_USD"],
                mode="markers",
                name=f"{joint_type} (listed)",
                marker=dict(
                    size=np.clip(type_data["Weight_kg"].fillna(0.4) * 40, 10, 28),
                    color=TYPE_COLORS.get(joint_type, "#00d4ff"),
                    line=dict(width=1, color="white"),
                    opacity=0.85,
                ),
                text=type_data.apply(
                    lambda r: (
                        f"<b>{r['Name']}</b><br>MFG: {r['MFG']}<br>"
                        f"Torque: {r['Torque_Nm']:.1f} Nm<br>"
                        f"Mass: {r['Weight_kg']:.2f} kg<br>"
                        f"Price: ${r['Cost_USD']:.0f}"
                    ),
                    axis=1,
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[torque],
            y=[cost],
            mode="markers+text",
            name="Your estimate",
            marker=dict(size=22, color="#ff6b6b", symbol="star", line=dict(width=2, color="white")),
            text=[f"${cost:,.0f}"],
            textposition="top center",
            textfont=dict(color="#ff6b6b", size=14),
            hovertemplate=(
                f"<b>YOUR JOINT</b><br>Type: {joint_type}<br>"
                f"Torque: {torque:.1f} Nm<br>Speed: {speed:.0f} rpm<br>"
                f"OD: {od:.0f} mm<br>Mass: {mass:.2f} kg<br>"
                f"Accuracy: {accuracy:.1f} arcmin<br>Est. ${cost:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{joint_type.upper()}</b> — Price vs Torque",
            font=dict(size=20, color="#00d4ff", family="JetBrains Mono"),
            x=0.5,
        ),
        xaxis=dict(
            title="Peak / stall torque (Nm)",
            type="log",
            tickfont=dict(color="#6b8ba4"),
            gridcolor="#2a4060",
            title_font=dict(color="#e8f4fc"),
        ),
        yaxis=dict(
            title="Price (USD)",
            type="log",
            tickfont=dict(color="#6b8ba4"),
            gridcolor="#2a4060",
            title_font=dict(color="#e8f4fc"),
        ),
        plot_bgcolor="#0a1628",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(18, 31, 54, 0.9)", bordercolor="#2a4060", font=dict(color="#e8f4fc")),
        height=560,
        margin=dict(l=60, r=40, t=80, b=60),
    )
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("### Your inputs")
        st.dataframe(
            pd.DataFrame(
                {
                    "Parameter": [
                        "Type",
                        "Peak torque",
                        "Rated speed",
                        "Outer diameter",
                        "Mass",
                        "Accuracy",
                        "Encoder",
                        "Brake / driver",
                    ],
                    "Value": [
                        joint_type,
                        f"{torque:.1f} Nm",
                        f"{speed:.0f} rpm",
                        f"{od:.0f} mm",
                        f"{mass:.2f} kg" + (" (est.)" if auto_mass else ""),
                        f"{accuracy:.1f} arcmin",
                        f"{encoder_bits} bit" + (" dual" if dual_encoder else " single"),
                        f"{'brake' if has_brake else 'no brake'} / {'driver' if has_driver else 'external drive'}",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("### Model quality")
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        "Priced joints in fit",
                        "Torque elasticity",
                        "Precision elasticity",
                        "In-sample R² (log price)",
                        "Leave-one-out R² (log price)",
                        "Leave-one-out MAE",
                        "Leave-one-out MAPE",
                    ],
                    "Value": [
                        str(stats["n_train"]),
                        f"{stats['torque_elasticity']:.2f}",
                        f"{stats['precision_elasticity']:.2f}",
                        f"{stats['r2_in_sample']:.2f}",
                        f"{stats['r2_loo']:.2f}",
                        f"${stats['mae_loo_usd']:.0f}",
                        f"{stats['mape_loo_pct']:.0f}%",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Accuracy is not in the CSV; the fit uses typical class backlash/repeatability "
            "unless you override it. Joint class (QDD vs harmonic vs series-elastic) is the "
            "main price driver. HEBI T-series lists at a flat $3000, so torque scaling inside "
            "that family is weak."
        )

    with st.expander("Priced joints used in the fit"):
        show = priced[
            ["Name", "MFG", "Type", "Torque_Nm", "Rated_Speed_rpm", "OD_mm", "Weight_kg", "Cost_USD"]
        ].sort_values("Cost_USD")
        st.dataframe(show, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
