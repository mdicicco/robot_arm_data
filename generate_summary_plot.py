"""
Generate summary visualizations of robot arm data.

Outputs:
  robot_arm_summary.png                    — README figure
  robot_arm_high_pf.png                    — PF > 0.4, reach < 2 m zoom
  robot_arm_human_comparison_hulls.png     — hulls + priced dots vs humans/humanoids,
                                             plus the actuator-only payload-ratio bound
                                             (arm_mass_model) across the humanoid payload range

Convention across all figures:
  - Convex hulls use every robot in a Type with mass/payload/reach
  - Scatter dots only for robots that also have price + repeatability
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon

import arm_mass_model as amodel
from scipy.spatial import ConvexHull

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'robot_arm_data.csv')
ref_path = os.path.join(script_dir, 'data', 'human_humanoid_arm_data.csv')

df_raw = pd.read_csv(data_path)
phys_cols = ['Reach_m', 'Weight_kg', 'Payload_kg']
value_cols = ['Cost_KUSD', 'Repeatability_mm', 'Reach_m', 'Weight_kg', 'Payload_kg']

# Hull set: every arm with physical axes (price not required)
df_hull = df_raw.dropna(subset=phys_cols).copy()
df_hull['Payload_Factor'] = df_hull['Payload_kg'] / df_hull['Weight_kg']

# Dot set: also needs price + repeatability (circle size = value metric)
df = df_raw.dropna(subset=value_cols).copy()
df['Payload_Factor'] = df['Payload_kg'] / df['Weight_kg']
df['value_metric'] = 1 / (df['Repeatability_mm'] * df['Cost_KUSD'])
value_min = df['value_metric'].min()
value_max = df['value_metric'].max()
df['marker_size'] = 30 + (df['value_metric'] - value_min) / (value_max - value_min) * 470

print(f"Robots in hulls (mass/payload/reach): {len(df_hull)}")
print(f"Robots with dots (price + repeatability): {len(df)}")
print(f"Types represented: {sorted(df_hull['Type'].unique().tolist())}")

unique_types = sorted(df_hull['Type'].unique())
color_palette = [
    '#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12',
    '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#8bc34a',
]
type_colors = {t: color_palette[i % len(color_palette)] for i, t in enumerate(unique_types)}
print(f"Color mapping: {type_colors}")

HUMAN_COLOR = '#f1c40f'
HUMANOID_COLOR = '#ff9f43'  # orange — unified humanoid family @ 5% body mass

# Annotation offsets for reference labels (name → (dx, dy) points)
REF_OFFSETS = {
    'Child': (10, -16),
    'Woman': (14, -28),
    'Man': (22, -14),
    'Strong': (10, 10),
    'G1': (10, -14),
    'G1 EDU': (10, 12),
    'R1': (-8, 12),
    'GR-1': (10, 8),
    'Optimus Gen2': (10, 10),
    'Figure 02': (10, -12),
    'Apollo': (10, 8),
    'Digit': (-10, -12),
    'Atlas': (10, 10),
}


def plot_robot_layers(
    ax,
    scatter_frame,
    hull_frame=None,
    alpha=0.7,
    with_hulls=True,
    with_labels=True,
    hull_alpha=0.15,
):
    """Draw type hulls from hull_frame; scatter dots from scatter_frame only.

    Hulls use every robot in a category with mass/payload/reach.
    Dots are sized by the frame's marker_size column (value metric on the
    priced figures; a constant on the humanoid figure).
    """
    if hull_frame is None:
        hull_frame = scatter_frame

    hull_types = sorted(hull_frame['Type'].unique())
    scatter_types = sorted(scatter_frame['Type'].unique())
    types = sorted(set(hull_types) | set(scatter_types))

    if with_hulls:
        for robot_type in types:
            type_df = hull_frame[hull_frame['Type'] == robot_type]
            color = type_colors.get(robot_type, '#6b8ba4')
            if len(type_df) >= 3:
                points = type_df[['Reach_m', 'Payload_Factor']].values
                try:
                    hull = ConvexHull(points)
                    ax.add_patch(Polygon(
                        points[hull.vertices],
                        alpha=hull_alpha,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=2,
                        zorder=2,
                    ))
                except Exception as e:
                    print(f"Could not create hull for {robot_type}: {e}")

    for robot_type in types:
        type_df = scatter_frame[scatter_frame['Type'] == robot_type]
        if type_df.empty:
            continue
        n_hull = len(hull_frame[hull_frame['Type'] == robot_type])
        label = None
        if with_labels:
            label = f"{robot_type.capitalize()} ({len(type_df)}/{n_hull})"
        ax.scatter(
            type_df['Reach_m'],
            type_df['Payload_Factor'],
            s=type_df['marker_size'],
            c=type_colors.get(robot_type, '#6b8ba4'),
            alpha=alpha,
            edgecolors='white',
            linewidths=0.5,
            label=label,
            zorder=3,
        )


def style_axes(ax, title, subtitle=None):
    ax.set_xlabel('Reach (m)', fontsize=14, color='#e8f4fc', fontweight='bold')
    ax.set_ylabel('Payload Factor (Payload / Robot Mass)', fontsize=14, color='#e8f4fc', fontweight='bold')
    ax.set_title(title, fontsize=18, color='#00d4ff', fontweight='bold')
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes, ha='center', fontsize=11,
            color='#6b8ba4', style='italic',
        )
    ax.grid(True, alpha=0.2, color='#2a4060')
    ax.tick_params(colors='#6b8ba4')
    for spine in ax.spines.values():
        spine.set_color('#2a4060')


def add_type_and_size_legends(ax, with_size=True, title='Robot Type', outside=False):
    legend = ax.legend(
        loc='upper left' if outside else 'upper right',
        bbox_to_anchor=(1.01, 1.0) if outside else None,
        fontsize=11,
        framealpha=0.9,
        facecolor='#121f36',
        edgecolor='#2a4060',
        labelcolor='#e8f4fc',
        title=title,
        title_fontsize=12,
    )
    legend.get_title().set_color('#00d4ff')
    if not with_size:
        return
    size_legend = ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#6b8ba4',
                   markersize=6, label='Lower value', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#6b8ba4',
                   markersize=18, label='Higher value', linestyle='None'),
        ],
        loc='lower right',
        fontsize=10,
        framealpha=0.9,
        facecolor='#121f36',
        edgecolor='#2a4060',
        labelcolor='#e8f4fc',
        title='Circle Size',
        title_fontsize=11,
    )
    size_legend.get_title().set_color('#00d4ff')
    ax.add_artist(legend)


def add_best_value_overlay(ax, best_rows):
    card = FancyBboxPatch(
        (0.705, 0.575), 0.275, 0.195,
        transform=ax.transAxes,
        boxstyle='round,pad=0.008,rounding_size=0.01',
        facecolor='#121f36',
        edgecolor='#2a4060',
        linewidth=1.0,
        alpha=0.94,
        zorder=6,
    )
    ax.add_patch(card)
    ax.text(
        0.718, 0.748, "Best value by type",
        transform=ax.transAxes, ha='left', va='center',
        fontsize=9, fontweight='bold', color='#00d4ff', zorder=7,
    )
    line_y = 0.718
    for robot_type, row in best_rows.iterrows():
        color = type_colors[robot_type]
        ax.scatter(
            [0.722], [line_y], s=22, c=color,
            edgecolors='white', linewidths=0.4,
            transform=ax.transAxes, zorder=7, clip_on=False,
        )
        ax.text(
            0.738, line_y,
            f"{row['Name']}  ${row['Cost_KUSD']:.2g}k  ±{row['Repeatability_mm']:g}mm",
            transform=ax.transAxes, ha='left', va='center',
            fontsize=8, color=color, zorder=7,
        )
        line_y -= 0.032


def plot_reference_group(ax, group_df, color, marker, label, size=160):
    """Draw hull + markers + name labels for human/humanoid refs."""
    if group_df.empty:
        return
    pts = group_df[['Reach_m', 'Payload_Factor']].values
    if len(group_df) >= 3:
        try:
            hull = ConvexHull(pts)
            ax.add_patch(Polygon(
                pts[hull.vertices],
                alpha=0.15,
                facecolor=color,
                edgecolor=color,
                linewidth=2,
                zorder=7,
            ))
        except Exception as e:
            print(f"Could not create hull for {label}: {e}")

    ax.scatter(
        group_df['Reach_m'],
        group_df['Payload_Factor'],
        s=size,
        marker=marker,
        c=color,
        edgecolors='white',
        linewidths=0.7,
        zorder=9,
        label=label,
    )
    for _, row in group_df.iterrows():
        offset = REF_OFFSETS.get(row['Name'], (10, 8))
        ax.annotate(
            row['Name'],
            (row['Reach_m'], row['Payload_Factor']),
            xytext=offset,
            textcoords='offset points',
            fontsize=9,
            color=color,
            fontweight='bold',
            ha='left' if offset[0] >= 0 else 'right',
            va='top' if offset[1] < 0 else 'bottom' if offset[1] > 0 else 'center',
            arrowprops=dict(arrowstyle='-', color=color, lw=0.8),
            zorder=11,
            annotation_clip=False,
        )


BOUND_COLOR = '#00d4ff'
BOUND_REACHES = np.linspace(0.25, 3.5, 66)


def humanoid_bound_curves(payloads, dof=7):
    """Actuator-only payload-ratio bound vs reach for each payload.

    Returns {payload: (full_arm_ratio, no_shoulder_ratio)} over BOUND_REACHES.
    'no_shoulder' drops the shoulder yaw + pitch actuators from the arm mass —
    on a humanoid they sit in the torso, so published arm mass excludes them.
    Uses the pooled "all" gearbox + motor fits and the model defaults
    (SF 1.0, off-chain joints at 50 %).
    """
    models = amodel.load_actuator_models()
    out = {}
    for payload in payloads:
        full, no_sh = [], []
        for reach in BOUND_REACHES:
            t = amodel.size_arm(amodel.ArmConfig(float(reach), float(payload), dof=dof), models)['table']
            full.append(payload / t['actuator_kg'].sum())
            no_sh.append(payload / t.loc[t['group'] != 'shoulder', 'actuator_kg'].sum())
        out[payload] = (np.array(full), np.array(no_sh))
    return out


def plot_humanoid_bound_band(ax, p_lo, p_hi, dof=7, y_max=None):
    """Shade the bound between the lightest and heaviest humanoid payloads."""
    curves = humanoid_bound_curves((p_lo, p_hi), dof=dof)
    y_max = y_max or ax.get_ylim()[1]
    x = BOUND_REACHES
    (lo_full, lo_ns), (hi_full, hi_ns) = curves[p_lo], curves[p_hi]
    ax.fill_between(x, lo_full, hi_full, color=BOUND_COLOR, alpha=0.10, zorder=3, linewidth=0)
    ax.fill_between(x, lo_ns, hi_ns, color=BOUND_COLOR, alpha=0.05, zorder=3, linewidth=0)
    labels = (
        f'Bound {p_lo:g}–{p_hi:g} kg, full {dof}-DOF arm',
        None,
        f'Bound {p_lo:g}–{p_hi:g} kg, shoulder actuators in torso',
        None,
    )
    for (y, ls, lw), label in zip(
        ((lo_full, '-', 1.8), (hi_full, '-', 1.8), (lo_ns, '--', 1.4), (hi_ns, '--', 1.4)), labels
    ):
        ax.plot(x, y, color=BOUND_COLOR, linestyle=ls, linewidth=lw, alpha=0.9, zorder=4, label=label)
    # Label each curve at the right edge of the visible x-range
    i_end = int(np.searchsorted(x, ax.get_xlim()[1], side='right')) - 1
    for y, text in ((lo_full, f'{p_lo:g} kg'), (hi_full, f'{p_hi:g} kg'),
                    (lo_ns, f'{p_lo:g} kg'), (hi_ns, f'{p_hi:g} kg')):
        ax.annotate(text, (x[i_end], y[i_end]), xytext=(-4, 6), textcoords='offset points',
                    ha='right', va='bottom', fontsize=8.5, color=BOUND_COLOR, zorder=4)
    return curves


# ---------------------------------------------------------------------------
# 1) README summary plot (robots only)
# ---------------------------------------------------------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#0a1628')
ax.set_facecolor('#0a1628')

plot_robot_layers(ax, df, hull_frame=df_hull)

best_rows = (
    df.sort_values('value_metric', ascending=False)
    .groupby('Type', sort=False)
    .head(1)
    .set_index('Type')
    .reindex(unique_types)
    .dropna(how='all')
)
for robot_type, row in best_rows.iterrows():
    ax.scatter(
        row['Reach_m'],
        row['Payload_Factor'],
        s=max(row['marker_size'] * 1.15, 80),
        facecolors='none',
        edgecolors=type_colors[robot_type],
        linewidths=2.4,
        zorder=5,
    )

style_axes(
    ax,
    'Robot Arm Comparison: Reach vs Payload Efficiency\n',
    'Hulls: all arms with mass/payload/reach  |  Dots: price + repeatability  |  Circle size = 1/(rep × price)',
)
add_type_and_size_legends(ax)

ax.text(
    0.02, 0.98,
    f"Dots: {len(df)}  |  Hulls: {len(df_hull)}  |  {len(unique_types)} types",
    transform=ax.transAxes, fontsize=10, color='#6b8ba4', va='top',
    bbox=dict(boxstyle='round', facecolor='#121f36', edgecolor='#2a4060', alpha=0.9),
)
add_best_value_overlay(ax, best_rows)

plt.tight_layout()
summary_path = os.path.join(script_dir, 'robot_arm_summary.png')
plt.savefig(summary_path, dpi=150, facecolor='#0a1628', edgecolor='none', bbox_inches='tight')
print(f"Saved plot to: {summary_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2) High-PF zoom — all phys-complete robots (no price/rep filter)
# ---------------------------------------------------------------------------
zoom = df_hull[(df_hull['Payload_Factor'] > 0.4) & (df_hull['Reach_m'] < 2.0)].copy()
zoom = zoom.sort_values('Payload_Factor', ascending=False)
# Prefer value-metric sizing when price+rep exist; otherwise a fixed size
zoom = zoom.merge(df[['Name', 'marker_size']], on='Name', how='left')
zoom['marker_size'] = zoom['marker_size'].fillna(80.0)

print(f"High-efficiency box (PF>0.4, reach<2m): {len(zoom)} robots (no price filter)")
print(zoom[['Name', 'MFG', 'Type', 'Payload_kg', 'Weight_kg', 'Payload_Factor', 'Reach_m']].to_string(index=False))

man_reach = float(
    pd.read_csv(ref_path).loc[lambda d: d['Name'] == 'Man', 'Reach_m'].iloc[0]
)

fig2, ax2 = plt.subplots(figsize=(12, 8))
fig2.patch.set_facecolor('#0a1628')
ax2.set_facecolor('#0a1628')
# Hulls + dots from the same high-PF set (price not required)
plot_robot_layers(ax2, zoom, hull_frame=zoom, alpha=0.85, with_hulls=True)

offsets = [
    (6, 8), (6, -10), (-6, 8), (-6, -10),
    (10, 2), (-12, 2), (8, -14), (-14, 10),
    (14, -6), (-8, 14), (4, 12), (-16, -6),
    (12, 10), (-10, -12), (16, 0),
]
for i, (_, row) in enumerate(zoom.iterrows()):
    dx, dy = offsets[i % len(offsets)]
    ax2.annotate(
        f"{row['Name']}",
        (row['Reach_m'], row['Payload_Factor']),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=8,
        color='#e8f4fc',
        ha='left' if dx >= 0 else 'right',
        arrowprops=dict(arrowstyle='-', color='#6b8ba4', lw=0.6),
        zorder=4,
    )

ax2.axvline(
    man_reach, color=HUMAN_COLOR, linestyle='--', linewidth=1.8,
    alpha=0.95, zorder=5, label=f'Man reach ({man_reach:g} m)',
)
y_hi = float(zoom['Payload_Factor'].max()) + 0.06
ax2.text(
    man_reach + 0.015, y_hi - 0.03, 'Man reach',
    color=HUMAN_COLOR, fontsize=10, fontweight='bold',
    ha='left', va='top', zorder=6,
)

x_lo = min(0.35, float(zoom['Reach_m'].min()) - 0.05)
x_hi = max(1.45, float(zoom['Reach_m'].max()) + 0.08)
ax2.set_xlim(x_lo, x_hi)
ax2.set_ylim(0.395, y_hi)
style_axes(
    ax2,
    'High payload-efficiency box  |  PF > 0.4, reach < 2 m',
    'All arms with mass/payload/reach (price not required)  |  Yellow dashed: adult male arm reach',
)
leg2 = ax2.legend(
    loc='upper right', fontsize=10, framealpha=0.9,
    facecolor='#121f36', edgecolor='#2a4060', labelcolor='#e8f4fc',
)
leg2.get_frame().set_alpha(0.9)
plt.tight_layout()
zoom_path = os.path.join(script_dir, 'robot_arm_high_pf.png')
fig2.savefig(zoom_path, dpi=150, facecolor='#0a1628', edgecolor='none', bbox_inches='tight')
plt.close(fig2)
print(f"Saved plot to: {zoom_path}")

# ---------------------------------------------------------------------------
# 3) Full-hull comparison + priced dots (humanoids unified @ 5% / carry÷4)
# ---------------------------------------------------------------------------
print(f"Hull-set robots (no price filter): {len(df_hull)}")

# Humanoids: one orange family — arm mass = 5% body;
# published whole-robot carry → per-arm full-reach ≈ carry/4 (÷2 arms, ÷2 close-body→reach)
CARRY_KG = {
    'Optimus Gen2': 20.0,
    'Figure 02': 20.0,
    'Apollo': 25.0,
    'Digit': 16.0,
    'Atlas': 30.0,
}
# Official one-arm ratings (already per-arm): apply only the full-reach ÷2 derate
PER_ARM_KG = {
    'G1': 2.0,
    'G1 EDU': 3.0,
    'R1': 2.0,
    'GR-1': 3.0,
}
HUMANOID_MASS_FRAC = 0.05

refs4 = pd.read_csv(ref_path)
humans4 = refs4[refs4['Type'] == 'human'].copy()
humans4['Payload_Factor'] = humans4['Payload_kg'] / humans4['Weight_kg']

h_rows = []
for _, row in refs4[refs4['Type'] == 'humanoid'].iterrows():
    body = row['Body_Mass_kg']
    if pd.isna(body) or body <= 0:
        continue
    name = row['Name']
    if name in CARRY_KG:
        payload = CARRY_KG[name] / 4.0
    elif name in PER_ARM_KG:
        payload = PER_ARM_KG[name] / 2.0  # full-reach derate only
    else:
        payload = float(row['Payload_kg']) / 2.0
    arm_mass = body * HUMANOID_MASS_FRAC
    h_rows.append({
        'Name': name,
        'Reach_m': row['Reach_m'],
        'Payload_kg': payload,
        'Weight_kg': arm_mass,
        'Payload_Factor': payload / arm_mass,
        'Body_Mass_kg': body,
    })
humanoids4 = pd.DataFrame(h_rows)

fig4, ax4 = plt.subplots(figsize=(18, 10))
fig4.patch.set_facecolor('#0a1628')
ax4.set_facecolor('#0a1628')

# Full-category hulls + dots only where price + repeatability exist
# Humanoid payload range drives both the dot filter and the bound band
H_PAYLOAD_LO = float(humanoids4['Payload_kg'].min())
H_PAYLOAD_HI = float(humanoids4['Payload_kg'].max())

# Hulls keep every arm; dots = every arm (priced or not) inside the humanoid
# payload range, one size, colored by type — no price / repeatability here
df_dots4 = df_hull[df_hull['Payload_kg'].between(H_PAYLOAD_LO, H_PAYLOAD_HI)].copy()
df_dots4['marker_size'] = 45
print(f"Humanoid-figure dots (payload {H_PAYLOAD_LO:g}–{H_PAYLOAD_HI:g} kg): {len(df_dots4)} of {len(df_hull)}")
plot_robot_layers(
    ax4, df_dots4, hull_frame=df_hull, alpha=0.7, with_hulls=True,
    with_labels=True, hull_alpha=0.22,
)

plot_reference_group(ax4, humans4, HUMAN_COLOR, '*', 'Human (full-reach ref.)', size=200)
plot_reference_group(
    ax4, humanoids4, HUMANOID_COLOR, '^',
    'Humanoid arm (5% body; carry÷4 or per-arm÷2)', size=130,
)

# Theoretical actuator-only bound across the humanoid payload range (7-DOF)
ax4.set_xlim(0, 3.0)
ax4.set_ylim(-0.05, 2.0)
plot_humanoid_bound_band(ax4, H_PAYLOAD_LO, H_PAYLOAD_HI, dof=7)

# Call out landmark peaks (Kinova now in research — keep LWR III + Mico 4)
for typ, name, offset, label, va in (
    ('research', 'LWR III', (14, -12), 'LWR III (1:1)', 'center'),
    ('research', 'Mico 4', (-12, 8), 'Mico 4', 'center'),
    ('collaborative', None, (-14, 6), None, 'center'),  # Z1 Pro — left of point
    ('hobby', 'myCobot 280', (-10, 10), 'myCobot 280', 'bottom'),  # above-left
):
    if name is None:
        sub = df_hull[df_hull['Type'] == typ]
        if sub.empty:
            continue
        row = sub.loc[sub['Payload_Factor'].idxmax()]
        label = row['Name']
    else:
        hit = df_hull[(df_hull['Name'] == name) & (df_hull['Type'] == typ)]
        if hit.empty:
            continue
        row = hit.iloc[0]
    color = type_colors.get(typ, '#6b8ba4')
    ax4.annotate(
        label,
        (row['Reach_m'], row['Payload_Factor']),
        xytext=offset,
        textcoords='offset points',
        fontsize=9,
        color=color,
        fontweight='bold',
        ha='right' if offset[0] < 0 else 'left',
        va=va,
        arrowprops=dict(arrowstyle='-', color=color, lw=0.8),
        zorder=9,
    )

style_axes(
    ax4,
    'Robot Arm Bounding Regions vs Human / Humanoid Arms\n',
    f'Hulls: all arms  |  Dots: all arms with payload {H_PAYLOAD_LO:g}–{H_PAYLOAD_HI:g} kg '
    '(legend n = dots/hull)  |  '
    'Orange ▲ humanoids @ 5% body mass  |  Cyan = actuator-only bound (no structure)',
)
add_type_and_size_legends(
    ax4, with_size=False, outside=True,
    title='Robot type  ·  references  ·\nactuator-only bound (SF 1.0, yaw/rolls 50 %)',
)

plt.tight_layout()
hull_path = os.path.join(script_dir, 'robot_arm_human_comparison_hulls.png')
fig4.savefig(hull_path, dpi=150, facecolor='#0a1628', edgecolor='none', bbox_inches='tight')
plt.close(fig4)
print(f"Saved plot to: {hull_path}")
print('Humanoid PF @ 5% / derated payload:')
print(humanoids4[['Name', 'Reach_m', 'Payload_kg', 'Weight_kg', 'Payload_Factor']].to_string(index=False))
