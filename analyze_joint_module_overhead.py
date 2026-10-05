"""
How much heavier is a complete joint module than its bare gearbox + motor?

Every row of data/robot_joint_data.csv is one single-DOF joint module. For each
one this reads the module's own torque / ratio / speed spec, runs it through the
fitted gearbox and motor mass models in arm_mass_model.py, and compares the
predicted gearbox + motor mass with the module's published mass:

    gearbox_kg = gearbox_fit(type, T_rated, ratio)
    motor_kg   = motor_fit(form, T_peak / (ratio · η[type]), rpm_rated · ratio)
    ratio      = module mass / (gearbox_kg + motor_kg)

This is the same per-module quantity arm_mass_model._calibrate_integration takes
the median of, so no integration factor is applied here. Structure, encoder,
driver and bearings are what the module adds on top of the bare components, so
the expectation is ratio > 1. Where the fits over-predict (ratio < 1) the
standalone catalog parts in the fits carry more packaging than the integrated
module does.

Model per module type:
    harmonic / cycloidal / planetary / qdd → type-matched gearbox fit + frameless motor
        (qdd uses the planetary gearbox fit, as in arm_mass_model.JOINT_TO_GEARBOX)
    series-elastic / hobby-servo           → pooled "all" gearbox + motor fits
The pooled fit is also computed for every module as a sensitivity column.

Outputs:
    robot_joint_module_overhead.png
    robot_joint_module_overhead.csv   (one row per module with a published mass)

Usage:
    python analyze_joint_module_overhead.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import arm_mass_model as amodel
from joint_price_model import load_joints

script_dir = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(script_dir, 'robot_joint_module_overhead.png')
OUT_CSV = os.path.join(script_dir, 'robot_joint_module_overhead.csv')

BG = '#0a1628'
PANEL = '#121f36'
GRID = '#2a4060'
INK = '#e8f4fc'
MUTED = '#6b8ba4'
TITLE = '#00d4ff'
# Polarity pair (validated all-pairs on BG): lighter than theory vs extra mass.
LIGHTER = '#3987e5'
HEAVIER = '#d95926'

# Type is carried by marker shape (color carries polarity), same shapes in both panels.
TYPE_MARKERS = {
    'harmonic': 'o',
    'qdd': 's',
    'planetary': '^',
    'cycloidal': 'D',
    'series-elastic': 'v',
    'hobby-servo': 'P',
}

DEFAULT_SPEED_RPM = 60.0   # same fallback as arm_mass_model._calibrate_integration
# A module counts as "outside the fit" only when a torque / ratio / speed is more than
# 25 % beyond the range the fit was trained on (so a 101:1 harmonic vs a 100:1 max is fine).
RANGE_MARGIN = 1.25


def component_mass(r: pd.Series, models: amodel.ActuatorModels, gear_key: str, motor_form: str) -> dict:
    """Bare gearbox + motor mass for one module, sized from its own spec.

    Missing rated/peak torque is filled from the other with PEAK_OVER_RATED, a
    missing ratio with the gearbox type's default, a missing speed with 60 rpm;
    each fill is reported so callers can flag it.
    """
    rated, peak = r['Rated_Torque_Nm'], r['Peak_Torque_Nm']
    torque_imputed = bool(pd.isna(rated) or pd.isna(peak))
    rated = peak / amodel.PEAK_OVER_RATED if pd.isna(rated) else float(rated)
    peak = rated * amodel.PEAK_OVER_RATED if pd.isna(peak) else float(peak)
    ratio_imputed = bool(pd.isna(r['Gear_Ratio']))
    ratio = amodel.DEFAULT_RATIO[gear_key] if ratio_imputed else float(r['Gear_Ratio'])
    speed_imputed = bool(pd.isna(r['Rated_Speed_rpm']))
    speed = DEFAULT_SPEED_RPM if speed_imputed else float(r['Rated_Speed_rpm'])

    motor_tau = peak / (ratio * amodel.GEAR_EFFICIENCY[gear_key])
    motor_rpm = speed * ratio
    gear_kg = models.gearbox_mass(gear_key, rated, ratio)
    motor_kg = models.motor_mass(motor_form, motor_tau, motor_rpm)

    g, m = models.gear_ranges.loc[gear_key], models.motor_ranges.loc[motor_form]
    tol = RANGE_MARGIN
    in_range = (
        g['t_min'] / tol <= rated <= g['t_max'] * tol and g['r_min'] / tol <= ratio <= g['r_max'] * tol
        and m['t_min'] / tol <= motor_tau <= m['t_max'] * tol and m['w_min'] / tol <= motor_rpm <= m['w_max'] * tol
    )
    return {
        'rated': rated, 'peak': peak, 'ratio': ratio, 'speed': speed,
        'torque_imputed': torque_imputed, 'ratio_imputed': ratio_imputed, 'speed_imputed': speed_imputed,
        'gear_kg': gear_kg, 'motor_kg': motor_kg, 'theory_kg': gear_kg + motor_kg,
        'extrapolated': not in_range,
    }


def module_table(joints: pd.DataFrame, models: amodel.ActuatorModels) -> pd.DataFrame:
    rows = []
    for _, r in joints.dropna(subset=['Weight_kg']).iterrows():
        gear_type = amodel.JOINT_TO_GEARBOX.get(r['Type'])
        typed = component_mass(r, models, gear_type or amodel.ALL, 'frameless' if gear_type else amodel.ALL)
        pooled = component_mass(r, models, amodel.ALL, amodel.ALL)
        actual = float(r['Weight_kg'])
        rows.append({
            'Name': r['Name'], 'MFG': r['MFG'], 'Type': r['Type'],
            'Module_kg': actual,
            'Model': f"{gear_type}+frameless" if gear_type else 'all+all',
            'Rated_Torque_Used_Nm': typed['rated'], 'Peak_Torque_Used_Nm': typed['peak'],
            'Ratio_Used': typed['ratio'], 'Speed_Used_rpm': typed['speed'],
            'Torque_Imputed': typed['torque_imputed'], 'Ratio_Imputed': typed['ratio_imputed'],
            'Speed_Imputed': typed['speed_imputed'],
            'Gearbox_kg': typed['gear_kg'], 'Motor_kg': typed['motor_kg'], 'Theory_kg': typed['theory_kg'],
            'Extra_kg': actual - typed['theory_kg'],
            'Mass_Ratio': actual / typed['theory_kg'],
            'Extrapolated': typed['extrapolated'],
            'Theory_Pooled_kg': pooled['theory_kg'],
            'Mass_Ratio_Pooled': actual / pooled['theory_kg'],
        })
    t = pd.DataFrame(rows)
    t['Any_Imputed'] = t[['Torque_Imputed', 'Ratio_Imputed', 'Speed_Imputed']].any(axis=1)
    return t


def type_summary(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby('Type')
    s = pd.DataFrame({
        'n': g.size(),
        'median_ratio': g['Mass_Ratio'].median(),
        'min_ratio': g['Mass_Ratio'].min(),
        'max_ratio': g['Mass_Ratio'].max(),
        'frac_lighter': g['Mass_Ratio'].apply(lambda x: float((x < 1).mean())),
        'median_ratio_pooled': g['Mass_Ratio_Pooled'].median(),
    })
    return s.sort_values('median_ratio')


def check_against_calibration(t: pd.DataFrame, models: amodel.ActuatorModels) -> None:
    """Per-type medians over rows with a published ratio should equal the model's integration factors."""
    pub = t[~t['Ratio_Imputed'] & t['Type'].isin(amodel.JOINT_TO_GEARBOX)].copy()
    pub['key'] = pub['Type'].map(amodel.JOINT_TO_GEARBOX)
    print('\nCheck vs arm_mass_model integration factors (published-ratio rows, type-matched fit):')
    for key, grp in pub.groupby('key'):
        print(f"  {key:10s} n={len(grp):3d}  median here {grp['Mass_Ratio'].median():.3f}   "
              f"models.integration {models.integration[key]:.3f}")
    print(f"  {'all':10s} n={len(pub):3d}  median here {pub['Mass_Ratio_Pooled'].median():.3f}   "
          f"models.integration {models.integration[amodel.ALL]:.3f}  (pooled fit)")


def style(ax) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.2, color=GRID)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(TITLE)


def draw_points(ax, x, y, frame: pd.DataFrame, size: float = 46) -> None:
    """One scatter call per (type, polarity, imputed) group: marker = type, color = polarity, hollow = imputed."""
    heavier = frame['Mass_Ratio'] >= 1
    for (typ, hv, imp), grp in frame.groupby(['Type', heavier, frame['Any_Imputed']]):
        color = HEAVIER if hv else LIGHTER
        idx = grp.index
        if imp:
            ax.scatter(x[idx], y[idx], s=size, marker=TYPE_MARKERS[typ], facecolors='none',
                       edgecolors=color, linewidths=1.4, zorder=3)
        else:
            ax.scatter(x[idx], y[idx], s=size, marker=TYPE_MARKERS[typ], facecolors=color,
                       edgecolors='white', linewidths=0.5, alpha=0.9, zorder=3)


def plot(t: pd.DataFrame, summary: pd.DataFrame, out: str) -> None:
    n = len(t)
    below = float((t['Mass_Ratio'] < 1).mean())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.4), gridspec_kw={'width_ratios': [1, 1.15]})
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        style(ax)

    # --- left: absolute masses -------------------------------------------------
    lo = min(t['Theory_kg'].min(), t['Module_kg'].min()) / 1.5
    hi = max(t['Theory_kg'].max(), t['Module_kg'].max()) * 1.5
    ax1.fill_between([lo, hi], [lo, hi], [hi, hi], color=HEAVIER, alpha=0.05, zorder=0)
    ax1.fill_between([lo, hi], [lo, lo], [lo, hi], color=LIGHTER, alpha=0.05, zorder=0)
    ax1.plot([lo, hi], [lo, hi], color=MUTED, linestyle='--', linewidth=1.2, zorder=1)
    draw_points(ax1, t['Theory_kg'], t['Module_kg'], t)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect('equal', adjustable='box')
    for axis in (ax1.xaxis, ax1.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax1.set_xlabel('Theoretical gearbox + motor mass from the fits (kg)')
    ax1.set_ylabel('Published module mass (kg)')
    ax1.set_title('Module mass vs. its components', fontsize=13, fontweight='bold', pad=8)
    ax1.text(0.03, 0.97, 'module heavier than theory\n(structure, encoder, driver)',
             transform=ax1.transAxes, color=MUTED, fontsize=9, va='top', ha='left')
    ax1.text(0.97, 0.03, 'module lighter than theory\n(fits over-predict)',
             transform=ax1.transAxes, color=MUTED, fontsize=9, va='bottom', ha='right')

    # --- right: ratio by type --------------------------------------------------
    order = list(summary.index)
    pos = {typ: i for i, typ in enumerate(order)}
    rng = np.random.default_rng(0)
    xj = t['Type'].map(pos).astype(float) + rng.uniform(-0.28, 0.28, size=n)
    ax2.axhspan(1, t['Mass_Ratio'].max() * 1.6, color=HEAVIER, alpha=0.05, zorder=0)
    ax2.axhspan(t['Mass_Ratio'].min() / 1.6, 1, color=LIGHTER, alpha=0.05, zorder=0)
    ax2.axhline(1, color=MUTED, linestyle='--', linewidth=1.2, zorder=1)
    draw_points(ax2, xj, t['Mass_Ratio'], t, size=40)
    for typ, i in pos.items():
        med = float(summary.loc[typ, 'median_ratio'])
        ax2.hlines(med, i - 0.36, i + 0.36, color=INK, linewidth=2.6, zorder=4)
    ax2.set_yscale('log')
    ax2.set_ylim(t['Mass_Ratio'].min() / 1.6, t['Mass_Ratio'].max() * 1.6)
    ticks = [v for v in (0.25, 0.5, 1, 2, 4) if ax2.get_ylim()[0] <= v <= ax2.get_ylim()[1]]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f'{v:g}×' for v in ticks])
    ax2.minorticks_off()
    ax2.set_xlim(-0.6, len(order) - 0.4)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(
        [f"{typ}\nn={int(summary.loc[typ, 'n'])}\nmedian {summary.loc[typ, 'median_ratio']:.2f}×\n"
         f"{summary.loc[typ, 'frac_lighter'] * 100:.0f}% lighter" for typ in order],
        fontsize=9, color=INK,
    )
    ax2.set_ylabel('Module mass ÷ theoretical gearbox + motor mass')
    ax2.set_title('Extra mass by module type (white bar = median)', fontsize=13, fontweight='bold', pad=8)

    handles = [Line2D([0], [0], marker=m, color='none', markerfacecolor=MUTED, markeredgecolor='white',
                      markeredgewidth=0.5, markersize=8, label=typ) for typ, m in TYPE_MARKERS.items()]
    handles += [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=HEAVIER, markeredgecolor='white',
               markeredgewidth=0.5, markersize=8, label='heavier than theory'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=LIGHTER, markeredgecolor='white',
               markeredgewidth=0.5, markersize=8, label='lighter than theory'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor=INK,
               markeredgewidth=1.4, markersize=8, label='hollow: an input was imputed'),
    ]
    # Right panel's upper-left is empty (only the heavier-than-theory types reach it)
    leg = ax2.legend(handles=handles, loc='upper left', fontsize=8.5,
                     facecolor=PANEL, edgecolor=GRID, labelcolor=INK, framealpha=0.95, title='marker = module type',
                     title_fontsize=9)
    leg.get_title().set_color(MUTED)
    leg.set_zorder(6)

    fig.suptitle(
        f'{below * 100:.0f}% of {n} joint modules weigh less than the bare gearbox + motor their torque spec predicts',
        fontsize=15, fontweight='bold', color=TITLE, y=0.99,
    )
    n_imp = int(t['Any_Imputed'].sum())
    n_ext = int(t['Extrapolated'].sum())
    fig.text(
        0.01, 0.005,
        'Theory = gearbox fit (rated torque, ratio) + motor fit (peak torque ÷ ratio·η, rated rpm × ratio) from arm_mass_model.py, '
        'no integration factor. Type-matched fits + frameless motor for harmonic / cycloidal / planetary / qdd; pooled fits for '
        'series-elastic / hobby-servo.\n'
        f'Hollow markers ({n_imp} of {n}): rated or peak torque filled with the 2.5× rule, ratio filled with the type default, '
        f'or speed with 60 rpm.  {n_ext} modules sit more than 25% outside the fitted torque / ratio / speed ranges '
        f'(CSV column Extrapolated).',
        color=MUTED, fontsize=8.5, ha='left', va='bottom',
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    models = amodel.load_actuator_models()
    joints = load_joints()
    no_mass = joints[joints['Weight_kg'].isna()]
    t = module_table(joints, models)
    print(f'{len(joints)} modules, {len(t)} with a published mass '
          f'(skipped, no mass: {", ".join(no_mass["Name"]) or "none"})')

    summary = type_summary(t)
    print('\nModule mass ÷ (theoretical gearbox + motor), by type:')
    print(summary.round(2).to_string())
    print(f"\nAll modules: median {t['Mass_Ratio'].median():.2f}×, "
          f"{(t['Mass_Ratio'] < 1).mean() * 100:.0f}% lighter than theory "
          f"({(t['Mass_Ratio_Pooled'] < 1).mean() * 100:.0f}% with the pooled fits)")
    clean = t[~t['Any_Imputed'] & ~t['Extrapolated']]
    print(f"Only rows with every input published and inside the fit ranges (n={len(clean)}): "
          f"median {clean['Mass_Ratio'].median():.2f}×, {(clean['Mass_Ratio'] < 1).mean() * 100:.0f}% lighter")
    check_against_calibration(t, models)

    out = t.copy()
    num = out.select_dtypes('number').columns
    out[num] = out[num].round(4)
    out.to_csv(OUT_CSV, index=False)
    print(f'\nWrote {OUT_CSV}')
    plot(t, summary, OUT_PNG)


if __name__ == '__main__':
    main()
