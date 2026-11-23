"""
Humanoid capability clusters on the height vs payload-factor plane.

Same axes as generate_humanoid_plot.py (X = height, Y = carry / body mass),
but every robot is coloured by one of four hand-defined groups:

  compact       height < 1.5 m (child-size / tabletop, low absolute carry)
  super         full-size, payload factor > 0.5 (claims above human carry)
  human-parity  full-size, 0.16 <= PF <= 0.5 (the cluster around the human star)
  low-carry     full-size, PF < 0.16 (heavy / wheeled / conservatively rated)

The thresholds sit in the empty gaps of the data: no robot between 1.40 and
1.60 m, none between PF 0.42 and 0.55, none between PF 0.14 and 0.18.

Outputs:
  humanoid_clusters.png
  per-group summary table on stdout
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'humanoid_robot_data.csv')
out_path = os.path.join(script_dir, 'humanoid_clusters.png')

COMPACT_MAX_H = 1.5
SUPER_MIN_PF = 0.5
LOW_MAX_PF = 0.16

HUMAN_HEIGHT_M, HUMAN_MASS_KG, HUMAN_CARRY_KG = 1.75, 78.0, 25.0
HUMAN_PF = HUMAN_CARRY_KG / HUMAN_MASS_KG

# key: (legend name, colour, canonical example)
GROUPS = {
    'parity':  ('Human-parity full-size', '#199e70', 'Figure 03'),
    'super':   ('Super-human carry claims', '#d95926', 'Atlas (Electric)'),
    'low':     ('Low-carry full-size', '#d55181', 'Galbot G1'),
    'compact': ('Compact / small', '#3987e5', 'Unitree G1'),
}

BG, FG, MUTED, GRID = '#0a1628', '#e8f4fc', '#6b8ba4', '#2a4060'


def assign_group(row):
    if row['Height_m'] < COMPACT_MAX_H:
        return 'compact'
    if row['Payload_Factor'] > SUPER_MIN_PF:
        return 'super'
    if row['Payload_Factor'] < LOW_MAX_PF:
        return 'low'
    return 'parity'


df = pd.read_csv(data_path)
plot = df.dropna(subset=['Height_m', 'Weight_kg', 'Payload_kg']).copy()
plot['Payload_Factor'] = plot['Payload_kg'] / plot['Weight_kg']
plot['Category'] = plot['Category'].fillna('commercial')
plot['Group'] = plot.apply(assign_group, axis=1)
m_min, m_max = plot['Weight_kg'].min(), plot['Weight_kg'].max()
plot['marker_size'] = 36 + (plot['Weight_kg'] - m_min) / (m_max - m_min) * 180

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 9.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Group hulls
for key, (_, color, _) in GROUPS.items():
    pts = plot.loc[plot['Group'] == key, ['Height_m', 'Payload_Factor']].values
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        ax.add_patch(Polygon(pts[hull.vertices], facecolor=color, edgecolor=color,
                             alpha=0.13, linewidth=1.8, zorder=1))

# Human reference
ax.axvline(HUMAN_HEIGHT_M, color=MUTED, linestyle=':', linewidth=1, alpha=0.6, zorder=0)
ax.axhline(HUMAN_PF, color=MUTED, linestyle='--', linewidth=1, alpha=0.6, zorder=0)
ax.scatter([HUMAN_HEIGHT_M], [HUMAN_PF], s=260, marker='*', c='white',
           edgecolors=BG, linewidths=0.8, zorder=6)

# Points: colour = group, diamond = research
for key, (_, color, canon) in GROUPS.items():
    g = plot[plot['Group'] == key]
    for cat, marker in (('commercial', 'o'), ('research', 'D')):
        s = g[g['Category'] == cat]
        ax.scatter(s['Height_m'], s['Payload_Factor'], s=s['marker_size'] * (0.8 if marker == 'D' else 1),
                   c=color, marker=marker, alpha=0.85, edgecolors='white', linewidths=0.7, zorder=3)
    c = g[g['Name'] == canon]
    ax.scatter(c['Height_m'], c['Payload_Factor'], s=c['marker_size'] * 2.6, facecolors='none',
               edgecolors=color, linewidths=2.2, zorder=4)

label_kw = dict(fontsize=8.5, va='center', zorder=5, annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6, shrinkA=0, shrinkB=4))


def label(row, xytext, coords, ha):
    canon = GROUPS[row['Group']][2] == row['Name']
    ax.annotate(row['Name'] + (' ★' if canon else ''), (row['Height_m'], row['Payload_Factor']),
                xytext=xytext, textcoords=coords, ha=ha,
                color=GROUPS[row['Group']][1] if canon else FG,
                fontweight='bold' if canon else 'normal', **label_kw)


def column(rows, x, top, gap, ha):
    rows = rows.sort_values(['Payload_Factor', 'Height_m'], ascending=[False, True])
    for i, (_, row) in enumerate(rows.iterrows()):
        label(row, (x, top - i * gap), 'data', ha)


# Dense full-size groups get stacked label columns; the rest get offsets.
GAP = 0.028
parity = plot[plot['Group'] == 'parity']
column(parity[parity['Height_m'] < 1.745], 1.55, 0.47, GAP, 'right')
column(parity[parity['Height_m'] >= 1.745], 1.97, 0.47, GAP, 'left')
low = plot[plot['Group'] == 'low']
column(low, 1.97, 0.47 - GAP * ((parity['Height_m'] >= 1.745).sum() + 0.6), GAP, 'left')

OFFSETS = {
    'Tuesday Labs Voxel Series-1': (4, 26), 'Fauna Sprout': (-4, -16), 'Noetix N2': (-8, 14),
    'Unitree R1': (-12, -14), 'UBTECH Walker X': (14, 14), 'ASIMO (2011)': (-12, 14),
    'Unitree G1': (12, -14), 'Unitree G1 EDU': (12, 14), 'K-Scale K-Bot': (-10, 16),
    'Booster T2': (-12, -4), '1X NEO': (12, 8), 'Fourier GR-1': (-14, 8),
    'Atlas (Electric)': (-14, 16),
}
for _, row in plot[plot['Group'].isin(['compact', 'super'])].iterrows():
    dx, dy = OFFSETS.get(row['Name'], (12, 10))
    label(row, (dx, dy), 'offset points', 'left' if dx >= 0 else 'right')


def summary(g):
    return pd.Series({
        'n': len(g),
        'height_m': g['Height_m'].median(),
        'mass_kg': g['Weight_kg'].median(),
        'carry_kg': g['Payload_kg'].median(),
        'PF': g['Payload_Factor'].median(),
        'DOF': g['Total_DOF'].median(),
        'battery_h': g['Battery_h'].median(),
        'speed_mps': g['Speed_mps'].median(),
        'price_k$': g['Cost_KUSD'].median(),
        'n_price': g['Cost_KUSD'].notna().sum(),
        'n_research': (g['Category'] == 'research').sum(),
    })


stats = plot.groupby('Group').apply(summary).loc[list(GROUPS)]

handles = []
for key, (name, color, canon) in GROUPS.items():
    s = stats.loc[key]
    handles.append(Line2D([0], [0], marker='o', linestyle='None', markersize=10,
                          markerfacecolor=color, markeredgecolor='white',
                          label=f"{name} (n={s['n']:.0f})  ·  {s['mass_kg']:.0f} kg, "
                                f"{s['carry_kg']:.0f} kg carry, PF {s['PF']:.2f}  ·  ★ {canon}"))
handles += [
    Line2D([0], [0], marker='D', linestyle='None', markersize=7, markerfacecolor=MUTED,
           markeredgecolor='white', label='diamond = research / open-source'),
    Line2D([0], [0], marker='*', linestyle='None', markersize=14, markerfacecolor='white',
           markeredgecolor=BG, label=f'Adult human (~{HUMAN_PF:.2f})'),
]
leg = ax.legend(handles=handles, loc='upper left', fontsize=9, framealpha=0.92,
                facecolor='#121f36', edgecolor=GRID, labelcolor=FG,
                title='Group (medians)  ·  ★ ringed = canonical example', title_fontsize=10)
leg.get_title().set_color('#00d4ff')

ax.set_xlabel('Height (m)', fontsize=13, color=FG, fontweight='bold')
ax.set_ylabel('Payload Factor (Carry / Body Mass)', fontsize=13, color=FG, fontweight='bold')
ax.set_title('Humanoid Capability Clusters  |  Height vs Carry / Mass',
             fontsize=16, color='#00d4ff', fontweight='bold', pad=10)
ax.grid(True, alpha=0.2, color=GRID)
ax.tick_params(colors=MUTED)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.set_xlim(0.80, 2.32)
ax.set_ylim(-0.06, float(plot['Payload_Factor'].max()) * 1.10)
ax.text(0.99, 0.01,
        f'compact < {COMPACT_MAX_H} m  ·  super PF > {SUPER_MIN_PF}  ·  low PF < {LOW_MAX_PF}  ·  '
        f'size = body mass ({m_min:.0f}–{m_max:.0f} kg)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color=MUTED)

plt.tight_layout()
fig.savefig(out_path, dpi=150, facecolor=BG, edgecolor='none', bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}\n')

pd.set_option('display.width', 200)
print(stats.round(2).to_string())
for key in GROUPS:
    names = plot.loc[plot['Group'] == key].sort_values('Payload_Factor', ascending=False)['Name']
    print(f'\n{key}: ' + ', '.join(names))
