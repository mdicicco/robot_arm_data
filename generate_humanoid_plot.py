"""
Generate humanoid hardware comparison figure.

One plot (arm-plot analog):
  X = height
  Y = payload factor = published carry / body mass
  Circle size = body mass
  Lines = generation progress within a product family

Outputs:
  humanoid_payload_efficiency.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from scipy.spatial import ConvexHull

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'humanoid_robot_data.csv')
out_path = os.path.join(script_dir, 'humanoid_payload_efficiency.png')
old_summary = os.path.join(script_dir, 'humanoid_summary.png')

df = pd.read_csv(data_path)
plot = df.dropna(subset=['Height_m', 'Weight_kg', 'Payload_kg']).copy()
plot['Payload_Factor'] = plot['Payload_kg'] / plot['Weight_kg']

m_min, m_max = plot['Weight_kg'].min(), plot['Weight_kg'].max()
plot['marker_size'] = 36 + (plot['Weight_kg'] - m_min) / (m_max - m_min) * 180

COLOR = '#ff9f43'
PROGRESS_COLOR = '#7ec8e3'

# Ordered generation chains (oldest → newest). Only names present on the
# height+mass+carry plot get connected.
PROGRESS_CHAINS = [
    ['Optimus Gen1', 'Optimus Gen2', 'Optimus Gen3'],
    ['Figure 01', 'Figure 02', 'Figure 03'],
    ['Digit', 'Digit v5'],
    ['Unitree H1', 'Unitree H1-2', 'Unitree H2'],
    ['Fourier GR-1', 'Fourier GR-2', 'Fourier GR-3'],
    ['UBTECH Walker S1', 'UBTECH Walker S2'],
    ['Unitree G1', 'Unitree G1 EDU'],
    ['Apollo', 'Apollo 2'],
    ['Atlas (Hydraulic)', 'Atlas (Electric)'],
]

# Label offsets in points (dx, dy). Mid-height cluster is fanned hard.
LABEL_OFFSETS = {
    'Fourier GR-1': (-28, 8),
    'Atlas (Electric)': (-18, 20),
    'ASIMO (2011)': (-18, 14),
    'Optimus Gen1': (-40, -8),
    'Optimus Gen2': (-6, 36),
    'Digit v5': (34, 30),
    'Apollo': (-42, 6),          # flipped to left
    'Figure 01': (-48, -6),
    'Figure 02': (-42, 16),
    'Figure 03': (36, -8),          # raised a bit so it clears the mid cluster
    'Unitree H1-2': (40, -8),
    'Digit': (-28, -28),         # flipped to left
    'Unitree H2': (30, -34),
    'UBTECH Walker S2': (-34, -34),
    'Unitree G1 EDU': (14, 20),
    'Unitree R1': (-16, 4),
    'Unitree G1': (14, -18),
}


def draw_progress_lines(ax, frame):
    """Connect successive generations; skip missing / coincident points."""
    by_name = frame.set_index('Name')
    n_segments = 0
    for chain in PROGRESS_CHAINS:
        present = [n for n in chain if n in by_name.index]
        for a, b in zip(present, present[1:]):
            xa, ya = float(by_name.loc[a, 'Height_m']), float(by_name.loc[a, 'Payload_Factor'])
            xb, yb = float(by_name.loc[b, 'Height_m']), float(by_name.loc[b, 'Payload_Factor'])
            if abs(xa - xb) < 1e-6 and abs(ya - yb) < 1e-6:
                # Same point on these axes (e.g. Figure 01→02) — skip zero-length
                continue
            arrow = FancyArrowPatch(
                (xa, ya), (xb, yb),
                arrowstyle='-|>', mutation_scale=12,
                color=PROGRESS_COLOR, linewidth=1.6, linestyle='-',
                alpha=0.85, zorder=2,
                shrinkA=6, shrinkB=6,
            )
            ax.add_patch(arrow)
            n_segments += 1
    return n_segments


plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor('#0a1628')
ax.set_facecolor('#0a1628')

pts = plot[['Height_m', 'Payload_Factor']].values
if len(pts) >= 3:
    try:
        hull = ConvexHull(pts)
        ax.add_patch(Polygon(
            pts[hull.vertices],
            alpha=0.18, facecolor=COLOR, edgecolor=COLOR,
            linewidth=2, zorder=1,
        ))
    except Exception as e:
        print(f'Hull skipped: {e}')

n_prog = draw_progress_lines(ax, plot)
print(f'Progress segments drawn: {n_prog}')

ax.scatter(
    plot['Height_m'], plot['Payload_Factor'],
    s=plot['marker_size'], c=COLOR,
    alpha=0.55, edgecolors='white', linewidths=0.7, zorder=3,
)

ax.axvspan(1.60, 1.85, color='#f1c40f', alpha=0.06, zorder=0)
ax.axvline(1.75, color='#f1c40f', linestyle=':', linewidth=1.2, alpha=0.75, zorder=2)
ax.text(
    1.755, -0.015, '≈ adult height',
    color='#f1c40f', fontsize=9, va='top', ha='left', zorder=4,
)

# Prefer labeling later gens when two gens share the same plotted point
label_names = set(plot['Name'])
# If Figure 01 and 02 coincide, keep both labels offset differently
for _, row in plot.iterrows():
    dx, dy = LABEL_OFFSETS.get(row['Name'], (14, 12))
    ax.annotate(
        row['Name'],
        (row['Height_m'], row['Payload_Factor']),
        xytext=(dx, dy), textcoords='offset points',
        fontsize=9, color='#e8f4fc', fontweight='bold',
        ha='left' if dx >= 0 else 'right',
        va='center',
        arrowprops=dict(arrowstyle='-', color='#6b8ba4', lw=0.75,
                        shrinkA=0, shrinkB=4),
        zorder=5,
        annotation_clip=False,
    )

ax.set_xlabel('Height (m)', fontsize=13, color='#e8f4fc', fontweight='bold')
ax.set_ylabel(
    'Payload Factor (Carry / Body Mass)',
    fontsize=13, color='#e8f4fc', fontweight='bold',
)
ax.set_title(
    'Humanoid Hardware  |  Height vs Carry / Mass',
    fontsize=16, color='#00d4ff', fontweight='bold', pad=10,
)

ax.grid(True, alpha=0.2, color='#2a4060')
ax.tick_params(colors='#6b8ba4')
for spine in ax.spines.values():
    spine.set_color('#2a4060')

y_max = float(plot['Payload_Factor'].max())
ax.set_ylim(-0.06, y_max * 1.20)
ax.set_xlim(float(plot['Height_m'].min()) - 0.16, float(plot['Height_m'].max()) + 0.18)

legend = ax.legend(
    handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR,
               markersize=5, label=f'~{m_min:.0f} kg', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR,
               markersize=11, label=f'~{m_max:.0f} kg', linestyle='None'),
        Line2D([0], [0], color=PROGRESS_COLOR, linewidth=2, label='Generation progress'),
    ],
    loc='lower right', fontsize=10, framealpha=0.9,
    facecolor='#121f36', edgecolor='#2a4060', labelcolor='#e8f4fc',
    title='Body mass (circle size)', title_fontsize=11,
)
legend.get_title().set_color('#00d4ff')

ax.text(
    0.02, 0.98,
    f'n={len(plot)}  ·  carry = published whole-robot / bimanual payload  ·  arrows = gen-to-gen',
    transform=ax.transAxes, fontsize=9, color='#6b8ba4',
    va='top', ha='left',
)

plt.tight_layout()
fig.savefig(out_path, dpi=150, facecolor='#0a1628', edgecolor='none', bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}')

if os.path.exists(old_summary):
    os.remove(old_summary)
    print(f'Removed superseded {old_summary}')

ordered = plot.sort_values('Payload_Factor', ascending=False)
print('\nRanked by carry/mass:')
print(
    ordered[['Name', 'Height_m', 'Weight_kg', 'Payload_kg', 'Payload_Factor']]
    .to_string(index=False)
)

# Show which chains contribute
print('\nProgress chains (on-plot members):')
by_name = set(plot['Name'])
for chain in PROGRESS_CHAINS:
    present = [n for n in chain if n in by_name]
    if len(present) >= 2:
        print('  ' + ' → '.join(present))
