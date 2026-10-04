"""
Humanoid analyses that lean on the arm data and the arm mass model.

1) humanoid_arm_actuator_share.png
   For every humanoid with a per-arm payload rating, size one arm with
   arm_mass_model (actuators only, no structure) at that payload and reach,
   and plot the minimum actuator mass as a share of body mass — with and
   without the shoulder yaw + pitch actuators (which a humanoid houses in the
   torso). A human arm is ~5 % of body mass, so a spec whose actuators alone
   approach 5 % leaves almost nothing for structure.

2) humanoid_vs_arm_payload_scaling.png
   Payload factor (payload / own mass) vs own mass on log-log axes for serial
   arms and whole humanoids, with a power-law fit per group. Arms lose payload
   factor slowly as they grow; humanoids sit flat near the human ~0.3.
   Ratings that aren't whole-body carry (HUMANOID_NOT_CARRY) are drawn hollow
   and kept out of the humanoid fit — with them in, three small Unitree
   single-arm ratings fake a strong positive slope.

Usage:
    python analyze_humanoid_scaling.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import arm_mass_model as amodel

script_dir = os.path.dirname(os.path.abspath(__file__))
HUMANOID_PATH = os.path.join(script_dir, 'data', 'humanoid_robot_data.csv')
ARM_PATH = os.path.join(script_dir, 'data', 'robot_arm_data.csv')
OUT_SHARE = os.path.join(script_dir, 'humanoid_arm_actuator_share.png')
OUT_SCALING = os.path.join(script_dir, 'humanoid_vs_arm_payload_scaling.png')

BG = '#0a1628'
PANEL = '#121f36'
GRID = '#2a4060'
INK = '#e8f4fc'
MUTED = '#6b8ba4'
TITLE = '#00d4ff'
HUMANOID_COLOR = '#ff9f43'
HUMAN_COLOR = '#f1c40f'
BOUND_COLOR = '#00d4ff'
# Same arm-type colors as the README arm figures (generate_summary_plot.py)
ARM_TYPE_COLORS = {
    'collaborative': '#2ecc71',
    'hobby': '#3498db',
    'industrial': '#e74c3c',
    'research': '#9b59b6',
}

HUMAN_ARM_SHARE = 0.05          # one human arm ≈ 5 % of body mass (de Leva segment sums)
ARM_REACH_OVER_HEIGHT = 0.44    # shoulder→hand when Arm_Reach_m is unpublished
HUMAN_MASS_KG, HUMAN_CARRY_KG = 78.0, 25.0   # same adult reference as generate_humanoid_plot.py

# Payload_kg values that are not comparable whole-body carry ratings
HUMANOID_NOT_CARRY = {
    'Unitree G1': 'single-arm rating',
    'Unitree G1 EDU': 'single-arm rating',
    'Unitree R1': 'single-arm rating',
    'Fourier GR-1': '40 kg = 6.7× the 2-arm rating',
}


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.2, color=GRID)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(TITLE)


def power_fit(x, y):
    """log y = a + b log x → (b, exp(a), R²)."""
    lx, ly = np.log(x), np.log(y)
    b, a = np.polyfit(lx, ly, 1)
    r2 = np.corrcoef(lx, ly)[0, 1] ** 2
    return float(b), float(np.exp(a)), float(r2)


# ---------------------------------------------------------------------------
# 1) Minimum arm-actuator share of body mass
# ---------------------------------------------------------------------------

def arm_actuator_shares(humanoids: pd.DataFrame) -> pd.DataFrame:
    models = amodel.load_actuator_models()
    rows = []
    for _, r in humanoids.dropna(subset=['Arm_Payload_kg', 'Weight_kg']).iterrows():
        reach_listed = pd.notna(r['Arm_Reach_m'])
        if reach_listed:
            reach = float(r['Arm_Reach_m'])
        elif pd.notna(r['Height_m']):
            reach = ARM_REACH_OVER_HEIGHT * float(r['Height_m'])
        else:
            continue
        # The model supports 6 or 7 DOF; 4–5 DOF arms are sized as 6 (one extra wrist roll)
        dof = 7 if pd.isna(r['Arm_DOF']) or r['Arm_DOF'] >= 7 else 6
        t = amodel.size_arm(
            amodel.ArmConfig(reach, float(r['Arm_Payload_kg']), dof=dof), models
        )['table']
        full = float(t['actuator_kg'].sum())
        no_shoulder = float(t.loc[t['group'] != 'shoulder', 'actuator_kg'].sum())
        body = float(r['Weight_kg'])
        rows.append({
            'Name': r['Name'],
            'Arm_Payload_kg': float(r['Arm_Payload_kg']),
            'Reach_m': reach,
            'Reach_listed': reach_listed,
            'Model_DOF': dof,
            'Body_kg': body,
            'Arm_full_kg': full,
            'Arm_no_shoulder_kg': no_shoulder,
            'Share_full': full / body,
            'Share_no_shoulder': no_shoulder / body,
        })
    return pd.DataFrame(rows).sort_values('Share_full').reset_index(drop=True)


def plot_share(share: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(share) + 2.4))
    fig.patch.set_facecolor(BG)
    style(ax)
    y = np.arange(len(share))

    ax.axvspan(0, HUMAN_ARM_SHARE * 100, color=HUMAN_COLOR, alpha=0.06, zorder=0)
    ax.axvline(HUMAN_ARM_SHARE * 100, color=HUMAN_COLOR, linestyle=':', linewidth=1.4, zorder=1)
    ax.text(HUMAN_ARM_SHARE * 100, len(share) - 0.35, ' human arm ≈ 5 % of body',
            color=HUMAN_COLOR, fontsize=9, va='bottom', ha='left')

    for i, r in share.iterrows():
        ax.plot([r['Share_no_shoulder'] * 100, r['Share_full'] * 100], [i, i],
                color=BOUND_COLOR, linewidth=2, alpha=0.55, zorder=2)
    ax.scatter(share['Share_no_shoulder'] * 100, y, s=70, facecolors=BG,
               edgecolors=BOUND_COLOR, linewidths=2, zorder=3)
    ax.scatter(share['Share_full'] * 100, y, s=70, color=BOUND_COLOR,
               edgecolors='white', linewidths=0.6, zorder=3)

    x_max = float(share['Share_full'].max()) * 100 * 1.08
    for i, r in share.iterrows():
        reach_txt = f"{r['Reach_m']:.2f} m" + ('' if r['Reach_listed'] else '*')
        ax.text(x_max * 1.01, i,
                f"{r['Arm_Payload_kg']:g} kg/arm @ {reach_txt} · {r['Body_kg']:g} kg body · "
                f"{r['Arm_full_kg']:.1f} kg/arm min",
                color=MUTED, fontsize=8.5, va='center', ha='left')

    ax.set_yticks(y)
    ax.set_yticklabels(share['Name'], fontsize=10, color=INK)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.7, len(share) - 0.1)
    ax.set_xlabel('Minimum actuator mass per arm, % of body mass (actuators only — no links, cabling, hand)')
    ax.set_title('Can the arm spec fit the body? Actuator-only arm mass at the rated per-arm payload',
                 fontsize=14, fontweight='bold', pad=10)

    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='none', markerfacecolor=BOUND_COLOR,
                   markeredgecolor='white', markersize=8, label='Whole arm incl. shoulder yaw + pitch'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=BG,
                   markeredgecolor=BOUND_COLOR, markeredgewidth=2, markersize=8,
                   label='Shoulder actuators in torso (elbow + wrist only)'),
        ],
        loc='lower right', fontsize=9.5, facecolor=PANEL, edgecolor=GRID, labelcolor=INK,
        framealpha=0.95,
    )
    leg.set_zorder(5)
    fig.text(
        0.01, 0.005,
        'arm_mass_model.py: pooled gearbox + motor fits, SF 1.0, yaw/rolls at 50 %, 7-DOF (6 if the arm has fewer '
        'joints).  * reach not published — 0.44 × height.',
        color=MUTED, fontsize=8.5, ha='left', va='bottom',
    )
    fig.tight_layout(rect=(0, 0.02, 0.78, 1))
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


# ---------------------------------------------------------------------------
# 2) Payload factor vs own mass — arms vs humanoids
# ---------------------------------------------------------------------------

def plot_scaling(arms: pd.DataFrame, hum: pd.DataFrame, out: str) -> dict:
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor(BG)
    style(ax)

    for typ, color in ARM_TYPE_COLORS.items():
        d = arms[arms['Type'] == typ]
        ax.scatter(d['Weight_kg'], d['PF'], s=22, color=color, alpha=0.55,
                   edgecolors='none', label=f'{typ.capitalize()} arm ({len(d)})', zorder=2)
    odd = hum['Name'].isin(HUMANOID_NOT_CARRY)
    clean, flagged = hum[~odd], hum[odd]
    ax.scatter(clean['Weight_kg'], clean['PF'], s=120, marker='^', color=HUMANOID_COLOR,
               edgecolors='white', linewidths=0.8, label=f'Humanoid, whole-body carry ({len(clean)})', zorder=4)
    ax.scatter(flagged['Weight_kg'], flagged['PF'], s=120, marker='^', facecolors='none',
               edgecolors=HUMANOID_COLOR, linewidths=1.6,
               label=f'Humanoid rating not comparable — excluded from fit ({len(flagged)})', zorder=4)
    ax.scatter([HUMAN_MASS_KG], [HUMAN_CARRY_KG / HUMAN_MASS_KG], s=260, marker='*', color=HUMAN_COLOR,
               edgecolors='white', linewidths=0.8, label='Adult human (~25 kg carry / 78 kg)', zorder=5)

    fits = {}
    for key, frame, color, lw in (
        ('arms', arms, '#a9b8c8', 2.0),
        ('cobots', arms[arms['Type'] == 'collaborative'], ARM_TYPE_COLORS['collaborative'], 1.6),
        ('humanoids', clean, HUMANOID_COLOR, 2.4),
    ):
        b, A, r2 = power_fit(frame['Weight_kg'].to_numpy(), frame['PF'].to_numpy())
        fits[key] = (b, A, r2, len(frame))
        xs = np.logspace(np.log10(frame['Weight_kg'].min()), np.log10(frame['Weight_kg'].max()), 50)
        name = {'arms': 'All arms', 'cobots': 'Cobots', 'humanoids': 'Humanoids (filled)'}[key]
        ax.plot(xs, A * xs ** b, color=color, linewidth=lw, linestyle='--' if key == 'cobots' else '-',
                zorder=3, label=f'{name}: PF ∝ mass^{b:+.2f}  (R² {r2:.2f})')

    for _, r in flagged.iterrows():
        ax.annotate(f"{r['Name']} ({HUMANOID_NOT_CARRY[r['Name']]})", (r['Weight_kg'], r['PF']),
                    xytext=(10, -4), textcoords='offset points', fontsize=8.5, color=HUMANOID_COLOR, zorder=6)
    for name, (dx, dy) in {'Atlas (Electric)': (10, 4), 'PNDbotics Adam SP': (10, -10),
                           'HRP-5P': (10, -2), 'Kepler K2 Bumblebee': (10, 6),
                           'Fauna Sprout': (-8, 10)}.items():
        hit = clean[clean['Name'] == name]
        if not hit.empty:
            r = hit.iloc[0]
            ax.annotate(name, (r['Weight_kg'], r['PF']), xytext=(dx, dy), textcoords='offset points',
                        ha='right' if dx < 0 else 'left',
                        fontsize=8.5, color=HUMANOID_COLOR, fontweight='bold', zorder=6)
    med = float(clean['PF'].median())
    ax.text(0.99, 0.985,
            f'Humanoid carry is whole-body, close to the body; arm payload is at full reach — not like-for-like.\n'
            f'Comparable humanoids: median PF {med:.2f}, IQR {clean["PF"].quantile(0.25):.2f}–'
            f'{clean["PF"].quantile(0.75):.2f}, across {clean["Weight_kg"].min():.0f}–{clean["Weight_kg"].max():.0f} kg.',
            transform=ax.transAxes, ha='right', va='top', fontsize=9, color=MUTED, zorder=7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Own mass (kg) — robot arm mass, or humanoid body mass')
    ax.set_ylabel('Payload factor (payload / own mass)')
    ax.set_title('Arms lose payload factor slowly as they grow; humanoids cluster near the human ~0.3',
                 fontsize=15, fontweight='bold', pad=10)
    leg = ax.legend(loc='lower left', fontsize=9.5, facecolor=PANEL, edgecolor=GRID,
                    labelcolor=INK, framealpha=0.92)
    leg.set_zorder(7)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')
    return fits


def main():
    hum_all = pd.read_csv(HUMANOID_PATH)

    share = arm_actuator_shares(hum_all)
    print('Minimum arm-actuator share of body mass (per arm):')
    print(share[['Name', 'Arm_Payload_kg', 'Reach_m', 'Body_kg', 'Arm_full_kg',
                 'Share_full', 'Share_no_shoulder']].round(3).to_string(index=False))
    plot_share(share, OUT_SHARE)

    arms = pd.read_csv(ARM_PATH).dropna(subset=['Payload_kg', 'Weight_kg'])
    arms = arms[(arms['Payload_kg'] > 0) & (arms['Weight_kg'] > 0)].copy()
    arms['PF'] = arms['Payload_kg'] / arms['Weight_kg']
    hum = hum_all.dropna(subset=['Payload_kg', 'Weight_kg']).copy()
    hum['PF'] = hum['Payload_kg'] / hum['Weight_kg']
    fits = plot_scaling(arms, hum, OUT_SCALING)
    print('\nPayload-factor scaling fits (PF ∝ mass^b):')
    for k, (b, A, r2, n) in fits.items():
        print(f'  {k:10s} b={b:+.2f}  R²={r2:.2f}  n={n}')


if __name__ == '__main__':
    main()
