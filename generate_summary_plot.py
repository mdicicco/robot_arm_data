"""
Generate a summary visualization of robot arm data.
Creates a scatter plot of Reach vs Payload Factor, colored by robot type,
with circle size based on value metric: 1 / (repeatability * price).
Includes convex hull shaded regions for each robot type.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'robot_arm_data.csv')
df = pd.read_csv(data_path)

# Only keep robots with ALL required fields: price, repeatability, reach, mass, payload
required_cols = ['Cost_KUSD', 'Repeatability_mm', 'Reach_m', 'Weight_kg', 'Payload_kg']
df = df.dropna(subset=required_cols)

print(f"Robots with complete data: {len(df)}")
print(f"Types represented: {df['Type'].unique().tolist()}")

# Calculate payload factor (payload / arm mass)
df['Payload_Factor'] = df['Payload_kg'] / df['Weight_kg']

# Calculate value metric: 1 / (repeatability * price)
# Higher value = better accuracy (lower repeatability) AND lower price
df['value_metric'] = 1 / (df['Repeatability_mm'] * df['Cost_KUSD'])

# Scale to reasonable marker sizes (30 to 500)
value_min = df['value_metric'].min()
value_max = df['value_metric'].max()
df['marker_size'] = 30 + (df['value_metric'] - value_min) / (value_max - value_min) * 470

# Distinct color for each robot type
unique_types = sorted(df['Type'].unique())
color_palette = [
    '#2ecc71',  # Green
    '#3498db',  # Blue
    '#e74c3c',  # Red
    '#9b59b6',  # Purple
    '#f39c12',  # Orange
    '#1abc9c',  # Teal
    '#e91e63',  # Pink
    '#00bcd4',  # Cyan
    '#ff5722',  # Deep Orange
    '#8bc34a',  # Light Green
]

type_colors = {t: color_palette[i % len(color_palette)] for i, t in enumerate(unique_types)}

print(f"Color mapping: {type_colors}")

# Create figure with dark background
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#0a1628')
ax.set_facecolor('#0a1628')

# Plot convex hulls first (so they appear behind points)
for robot_type in unique_types:
    type_df = df[df['Type'] == robot_type]
    color = type_colors[robot_type]
    
    # Need at least 3 points for a convex hull
    if len(type_df) >= 3:
        points = type_df[['Reach_m', 'Payload_Factor']].values
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            polygon = Polygon(hull_points, alpha=0.15, facecolor=color, edgecolor=color, linewidth=2)
            ax.add_patch(polygon)
        except Exception as e:
            print(f"Could not create hull for {robot_type}: {e}")

# Plot each type separately for legend
for robot_type in unique_types:
    type_df = df[df['Type'] == robot_type]
    color = type_colors[robot_type]
    
    ax.scatter(
        type_df['Reach_m'],
        type_df['Payload_Factor'],
        s=type_df['marker_size'],
        c=color,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5,
        label=f"{robot_type.capitalize()} ({len(type_df)})"
    )

# Labels and title
ax.set_xlabel('Reach (m)', fontsize=14, color='#e8f4fc', fontweight='bold')
ax.set_ylabel('Payload Factor (Payload / Robot Mass)', fontsize=14, color='#e8f4fc', fontweight='bold')
ax.set_title('Robot Arm Comparison: Reach vs Payload Efficiency\n', 
             fontsize=18, color='#00d4ff', fontweight='bold')

# Subtitle
ax.text(0.5, 1.02, 'Circle size: value metric = 1 / (repeatability × price)  |  Larger = better value',
        transform=ax.transAxes, ha='center', fontsize=11, color='#6b8ba4', style='italic')

# Grid styling
ax.grid(True, alpha=0.2, color='#2a4060')
ax.tick_params(colors='#6b8ba4')

# Spines styling
for spine in ax.spines.values():
    spine.set_color('#2a4060')

# Legend for types
legend = ax.legend(
    loc='upper right',
    fontsize=11,
    framealpha=0.9,
    facecolor='#121f36',
    edgecolor='#2a4060',
    labelcolor='#e8f4fc',
    title='Robot Type',
    title_fontsize=12
)
legend.get_title().set_color('#00d4ff')

# Add size legend
size_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#6b8ba4', 
           markersize=6, label='Lower value', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#6b8ba4', 
           markersize=18, label='Higher value', linestyle='None'),
]
size_legend = ax.legend(
    handles=size_legend_elements,
    loc='lower right',
    fontsize=10,
    framealpha=0.9,
    facecolor='#121f36',
    edgecolor='#2a4060',
    labelcolor='#e8f4fc',
    title='Circle Size',
    title_fontsize=11
)
size_legend.get_title().set_color('#00d4ff')
ax.add_artist(legend)  # Re-add the first legend

# Stats annotation
total_robots = len(df)
types_count = len(unique_types)
stats_text = f"Total: {total_robots} robots | {types_count} types (complete data only)"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        fontsize=10, color='#6b8ba4', va='top',
        bbox=dict(boxstyle='round', facecolor='#121f36', edgecolor='#2a4060', alpha=0.9))

plt.tight_layout()

# Save the figure
output_path = os.path.join(script_dir, 'robot_arm_summary.png')
plt.savefig(output_path, dpi=150, facecolor='#0a1628', edgecolor='none', bbox_inches='tight')
print(f"Saved plot to: {output_path}")
