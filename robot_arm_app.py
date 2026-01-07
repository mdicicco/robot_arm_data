"""
Interactive Robot Arm Analysis App
Modern web-based GUI using Streamlit with interactive Plotly charts.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Robot Arm Analyzer",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark industrial theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2d4a 50%, #0a1628 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121f36 0%, #0a1628 100%);
        border-right: 2px solid #2a4060;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a2d4a, #121f36);
        border: 1px solid #2a4060;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
    }
    
    div[data-testid="stMetricValue"] {
        color: #7bed9f !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #6b8ba4 !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #e8f4fc !important;
        font-weight: 500;
    }
    
    /* Select boxes */
    .stSelectbox label {
        color: #00d4ff !important;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(26, 45, 74, 0.8);
        border: 1px solid #2a4060;
    }
    
    /* Divider */
    hr {
        border-color: #2a4060;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Load robot arm data from CSV."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data', 'robot_arm_data.csv')
    return pd.read_csv(file_path)


@st.cache_resource
def train_models(data):
    """Train regression models for mass and cost prediction."""
    # Clean data for modeling
    model_data = data.dropna(
        subset=['Weight_kg', 'Cost_KUSD', 'Payload_kg', 'Reach_m', 'Repeatability_mm', 'DOF']
    )
    
    features = ['Payload_kg', 'Reach_m', 'Repeatability_mm', 'DOF']
    X = model_data[features]
    y_mass = model_data['Weight_kg']
    y_cost = model_data['Cost_KUSD']
    
    model_mass = LinearRegression()
    model_mass.fit(X, y_mass)
    
    model_cost = LinearRegression()
    model_cost.fit(X, y_cost)
    
    return model_mass, model_cost, features


def predict_values(model_mass, model_cost, features, payload, reach, repeatability, dof):
    """Predict mass and cost from parameters."""
    X = pd.DataFrame([[payload, reach, repeatability, dof]], columns=features)
    mass = max(0.1, model_mass.predict(X)[0])
    cost = max(0.1, model_cost.predict(X)[0])
    return mass, cost


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load data and train models
    raw_data = load_data()
    model_mass, model_cost, features = train_models(raw_data)
    
    # Get unique robot types
    robot_types = sorted(raw_data['Type'].dropna().unique().tolist())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIDEBAR - Controls
    # ═══════════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.markdown("# 🦾 Robot Config")
        st.markdown("---")
        
        # Robot Type Selector
        st.markdown("### 📋 Robot Type")
        robot_type = st.selectbox(
            "Select category",
            robot_types,
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Parameters")
        
        # DOF Slider
        dof = st.slider(
            "Degrees of Freedom (DOF)",
            min_value=int(raw_data['DOF'].min()),
            max_value=int(raw_data['DOF'].max()),
            value=6,
            step=1
        )
        
        # Reach Slider
        reach = st.slider(
            "Reach (m)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.01,
            format="%.2f"
        )
        
        # Payload Slider
        payload = st.slider(
            "Payload (kg)",
            min_value=0.05,
            max_value=500.0,
            value=5.0,
            step=0.5,
            format="%.1f"
        )
        
        # Repeatability Slider
        repeatability = st.slider(
            "Repeatability (mm)",
            min_value=0.005,
            max_value=5.0,
            value=0.1,
            step=0.005,
            format="%.3f"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Mass Settings")
        
        # Auto-estimate mass toggle
        auto_mass = st.toggle("Auto-estimate mass", value=True)
        
        if auto_mass:
            est_mass, _ = predict_values(model_mass, model_cost, features, payload, reach, repeatability, dof)
            mass = est_mass
            st.info(f"Estimated: **{mass:.1f} kg**")
        else:
            mass = st.slider(
                "Arm Mass (kg)",
                min_value=1.0,
                max_value=1000.0,
                value=30.0,
                step=1.0,
                format="%.1f"
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN CONTENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Header
    st.markdown("# 🤖 Robot Arm Analysis")
    st.markdown("*Interactive payload factor analysis with cost estimation*")
    
    # Calculate predictions
    _, cost = predict_values(model_mass, model_cost, features, payload, reach, repeatability, dof)
    payload_factor = payload / mass if mass > 0 else 0
    price_per_dof = cost / dof if dof > 0 else 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # METRICS ROW
    # ═══════════════════════════════════════════════════════════════════════════
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Cost",
            value=f"${cost:.2f}K",
            delta=None
        )
    
    with col2:
        st.metric(
            label="📊 Price per DOF",
            value=f"${price_per_dof:.2f}K",
            delta=None
        )
    
    with col3:
        st.metric(
            label="⚖️ Robot Mass",
            value=f"{mass:.1f} kg",
            delta="estimated" if auto_mass else "manual"
        )
    
    with col4:
        st.metric(
            label="💪 Payload Factor",
            value=f"{payload_factor:.3f}",
            delta=None
        )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN CHART
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Filter data by robot type
    type_data = raw_data[raw_data['Type'] == robot_type].copy()
    type_data = type_data.dropna(subset=['Payload_kg', 'Weight_kg', 'Reach_m'])
    type_data['Payload_Factor'] = type_data['Payload_kg'] / type_data['Weight_kg']
    type_data['Has_Cost'] = type_data['Cost_KUSD'].notna()
    
    # Create the main scatter plot
    fig = go.Figure()
    
    # Add existing robots - with cost data
    robots_with_cost = type_data[type_data['Has_Cost']]
    if len(robots_with_cost) > 0:
        fig.add_trace(go.Scatter(
            x=robots_with_cost['Reach_m'],
            y=robots_with_cost['Payload_Factor'],
            mode='markers',
            name='Robots (with cost data)',
            marker=dict(
                size=robots_with_cost['Payload_kg'] / robots_with_cost['Payload_kg'].max() * 40 + 10,
                color=robots_with_cost['Cost_KUSD'],
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(
                        text="Cost ($K)",
                        side="right",
                        font=dict(color='#e8f4fc')
                    ),
                    tickfont=dict(color='#e8f4fc')
                ),
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=robots_with_cost.apply(
                lambda r: f"<b>{r['Name']}</b><br>"
                         f"MFG: {r['MFG']}<br>"
                         f"Payload: {r['Payload_kg']:.1f} kg<br>"
                         f"Reach: {r['Reach_m']:.2f} m<br>"
                         f"Mass: {r['Weight_kg']:.1f} kg<br>"
                         f"Cost: ${r['Cost_KUSD']:.1f}K<br>"
                         f"DOF: {r['DOF']}",
                axis=1
            ),
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add existing robots - without cost data (dimmed)
    robots_no_cost = type_data[~type_data['Has_Cost']]
    if len(robots_no_cost) > 0:
        fig.add_trace(go.Scatter(
            x=robots_no_cost['Reach_m'],
            y=robots_no_cost['Payload_Factor'],
            mode='markers',
            name='Robots (no cost data)',
            marker=dict(
                size=robots_no_cost['Payload_kg'] / type_data['Payload_kg'].max() * 40 + 8,
                color='#4a6080',
                line=dict(width=1, color='#6b8ba4'),
                opacity=0.5
            ),
            text=robots_no_cost.apply(
                lambda r: f"<b>{r['Name']}</b><br>"
                         f"MFG: {r['MFG']}<br>"
                         f"Payload: {r['Payload_kg']:.1f} kg<br>"
                         f"Reach: {r['Reach_m']:.2f} m<br>"
                         f"Mass: {r['Weight_kg']:.1f} kg<br>"
                         f"DOF: {r['DOF']}<br>"
                         f"<i>No cost data</i>",
                axis=1
            ),
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add user configuration as a star
    fig.add_trace(go.Scatter(
        x=[reach],
        y=[payload_factor],
        mode='markers+text',
        name='Your Configuration',
        marker=dict(
            size=25,
            color='#ff6b6b',
            symbol='star',
            line=dict(width=2, color='white')
        ),
        text=[f"${cost:.1f}K"],
        textposition="top center",
        textfont=dict(color='#ff6b6b', size=14, family='JetBrains Mono'),
        hovertemplate=(
            f"<b>YOUR CONFIG</b><br>"
            f"Reach: {reach:.2f} m<br>"
            f"Payload: {payload:.1f} kg<br>"
            f"Mass: {mass:.1f} kg<br>"
            f"Payload Factor: {payload_factor:.3f}<br>"
            f"Est. Cost: ${cost:.1f}K<br>"
            f"DOF: {dof}<extra></extra>"
        )
    ))
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text=f"<b>{robot_type.upper()} ROBOTS</b> — Payload Factor vs Reach",
            font=dict(size=20, color='#00d4ff', family='JetBrains Mono'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Reach (m)", font=dict(size=14, color='#e8f4fc')),
            tickfont=dict(color='#6b8ba4'),
            gridcolor='#2a4060',
            zerolinecolor='#2a4060'
        ),
        yaxis=dict(
            title=dict(text="Payload Factor (Payload / Robot Mass)", font=dict(size=14, color='#e8f4fc')),
            tickfont=dict(color='#6b8ba4'),
            gridcolor='#2a4060',
            zerolinecolor='#2a4060'
        ),
        plot_bgcolor='#0a1628',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            bgcolor='rgba(18, 31, 54, 0.9)',
            bordercolor='#2a4060',
            font=dict(color='#e8f4fc'),
            x=0.02,
            y=0.98
        ),
        height=600,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DETAILS SECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📈 Your Configuration Details")
        
        config_df = pd.DataFrame({
            'Parameter': ['DOF', 'Reach', 'Payload', 'Mass', 'Repeatability', 'Payload Factor'],
            'Value': [
                f"{dof}",
                f"{reach:.2f} m",
                f"{payload:.1f} kg",
                f"{mass:.1f} kg {'(est.)' if auto_mass else ''}",
                f"{repeatability:.3f} mm",
                f"{payload_factor:.4f}"
            ]
        })
        st.dataframe(config_df, hide_index=True, width="stretch")
    
    with col_right:
        st.markdown("### 📊 Dataset Statistics")
        
        stats_df = pd.DataFrame({
            'Metric': [
                f'Total {robot_type} robots',
                'With cost data',
                'Avg. Payload Factor',
                'Avg. Cost (where known)'
            ],
            'Value': [
                str(len(type_data)),
                str(len(robots_with_cost)),
                f"{type_data['Payload_Factor'].mean():.3f}" if len(type_data) > 0 else "N/A",
                f"${robots_with_cost['Cost_KUSD'].mean():.1f}K" if len(robots_with_cost) > 0 else "N/A"
            ]
        })
        st.dataframe(stats_df, hide_index=True, width="stretch")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    with st.expander("📋 View Raw Data Table"):
        display_cols = ['Name', 'MFG', 'DOF', 'Payload_kg', 'Reach_m', 'Weight_kg', 
                       'Repeatability_mm', 'Cost_KUSD', 'Payload_Factor']
        display_data = type_data[display_cols].sort_values('Payload_Factor', ascending=False)
        st.dataframe(display_data, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()

