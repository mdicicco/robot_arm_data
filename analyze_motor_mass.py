"""
Fit a BLDC motor mass model from peak torque and max speed.

    log(mass) = a + b · log(max_torque) + c · log(max_speed)

Usage:
    python analyze_motor_mass.py
    python analyze_motor_mass.py --torque 2.0 --speed 4000
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

FORM_COLORS = {
    "frameless": "#2ecc71",
    "flat": "#3498db",
    "inrunner": "#9b59b6",
    "outrunner": "#e67e22",
    "industrial": "#e74c3c",
    "integrated": "#1abc9c",
    "hub": "#f1c40f",
}


def data_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bldc_motor_data.csv")


def load_motors(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or data_path())


def train_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with listed mass + max torque + max speed."""
    out = df.copy()
    out = out.dropna(subset=["Max_Torque_Nm", "Max_Speed_rpm", "Weight_kg"])
    out = out[out["Weight_kg"] > 0]
    out = out[out["Max_Torque_Nm"] > 0]
    out = out[out["Max_Speed_rpm"] > 0]
    # Prefer manufacturer/shop masses for the fit
    if "Weight_Flag" in out.columns:
        listed = out[out["Weight_Flag"].fillna("listed") == "listed"]
        if len(listed) >= 8:
            out = listed
    out = out.copy()
    out["log_torque"] = np.log(out["Max_Torque_Nm"].astype(float))
    out["log_speed"] = np.log(out["Max_Speed_rpm"].astype(float))
    out["log_mass"] = np.log(out["Weight_kg"].astype(float))
    return out


def fit_mass_model(train: pd.DataFrame):
    X = train[["log_torque", "log_speed"]].to_numpy()
    y = train["log_mass"].to_numpy()
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    loo = LeaveOneOut()
    y_loo = cross_val_predict(LinearRegression(), X, y, cv=loo)
    return {
        "model": model,
        "a": float(model.intercept_),
        "b_torque": float(model.coef_[0]),
        "c_speed": float(model.coef_[1]),
        "r2": float(r2_score(y, y_hat)),
        "r2_loo": float(r2_score(y, y_loo)),
        "mae_kg": float(mean_absolute_error(np.exp(y), np.exp(y_hat))),
        "mae_loo_kg": float(mean_absolute_error(np.exp(y), np.exp(y_loo))),
        "y_hat": y_hat,
        "y_loo": y_loo,
        "n": len(train),
    }


def estimate_mass(model, torque_nm: float, speed_rpm: float) -> float:
    log_m = model.predict([[np.log(torque_nm), np.log(speed_rpm)]])[0]
    return float(np.exp(log_m))


def _style_axes(ax):
    ax.set_facecolor("#0a1628")
    ax.grid(True, alpha=0.2, color="#2a4060")
    ax.tick_params(colors="#6b8ba4")
    for spine in ax.spines.values():
        spine.set_color("#2a4060")
    ax.xaxis.label.set_color("#e8f4fc")
    ax.yaxis.label.set_color("#e8f4fc")
    ax.title.set_color("#00d4ff")


def plot_fit(train: pd.DataFrame, fit: dict, output_path: str) -> None:
    train = train.copy()
    train["pred_kg"] = np.exp(fit["y_hat"])
    train["loo_kg"] = np.exp(fit["y_loo"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#0a1628")

    # Left: mass vs peak torque, color by form, size ~ 1/speed
    ax = axes[0]
    _style_axes(ax)
    for form, color in FORM_COLORS.items():
        subset = train[train["Form"] == form]
        if subset.empty:
            continue
        sizes = np.clip(8000.0 / subset["Max_Speed_rpm"].astype(float), 25, 320)
        ax.scatter(
            subset["Max_Torque_Nm"],
            subset["Weight_kg"],
            s=sizes,
            c=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            label=f"{form} ({len(subset)})",
            zorder=3,
        )
    # Iso-speed contour sketch from the fitted surface at 3k / 8k rpm
    t = np.logspace(np.log10(train["Max_Torque_Nm"].min() * 0.8),
                    np.log10(train["Max_Torque_Nm"].max() * 1.2), 80)
    for spd, style in [(3000, "--"), (8000, ":")]:
        m = [estimate_mass(fit["model"], ti, spd) for ti in t]
        ax.plot(t, m, style, color="#6b8ba4", linewidth=1.2, alpha=0.9,
                label=f"model @ {spd} rpm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Max / peak torque (Nm)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("BLDC mass vs torque (size ∝ 1/speed)")
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        facecolor="#121f36",
        edgecolor="#2a4060",
        labelcolor="#e8f4fc",
    )

    # Right: predicted vs actual
    ax = axes[1]
    _style_axes(ax)
    lo = min(train["Weight_kg"].min(), train["pred_kg"].min()) * 0.7
    hi = max(train["Weight_kg"].max(), train["pred_kg"].max()) * 1.3
    ax.plot([lo, hi], [lo, hi], color="#6b8ba4", linestyle="--", linewidth=1, zorder=1)
    for form, color in FORM_COLORS.items():
        subset = train[train["Form"] == form]
        if subset.empty:
            continue
        ax.scatter(
            subset["Weight_kg"],
            subset["pred_kg"],
            s=55,
            c=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual mass (kg)")
    ax.set_ylabel("Predicted mass (kg)")
    ax.set_title(
        f"log(m)=a+b·log(τ)+c·log(ω)  |  R²={fit['r2']:.2f}  LOO R²={fit['r2_loo']:.2f}"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(
        0.02,
        0.98,
        f"n={fit['n']}  MAE={fit['mae_kg']:.2f} kg  LOO MAE={fit['mae_loo_kg']:.2f} kg\n"
        f"a={fit['a']:.3f}  b_τ={fit['b_torque']:.3f}  c_ω={fit['c_speed']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#e8f4fc",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#121f36", edgecolor="#2a4060"),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_contour(train: pd.DataFrame, fit: dict, output_path: str) -> None:
    """2D torque–speed map: filled contours = model mass, dots = measured mass."""
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.patch.set_facecolor("#0a1628")
    _style_axes(ax)

    t_lo = train["Max_Torque_Nm"].min() * 0.7
    t_hi = train["Max_Torque_Nm"].max() * 1.15
    # Keep the ultra-high-speed Faulhaber visible but don't dominate the grid
    s_lo = max(train["Max_Speed_rpm"].min() * 0.85, 300)
    s_hi = min(train["Max_Speed_rpm"].max() * 1.05, 25000)

    tt = np.logspace(np.log10(t_lo), np.log10(t_hi), 120)
    ss = np.logspace(np.log10(s_lo), np.log10(s_hi), 120)
    T, S = np.meshgrid(tt, ss)
    M = np.exp(
        fit["a"]
        + fit["b_torque"] * np.log(T)
        + fit["c_speed"] * np.log(S)
    )

    # Shared log color scale from data + model in-plot range
    m_data = train["Weight_kg"].astype(float)
    vmin = min(m_data.min(), float(M.min())) * 0.9
    vmax = max(m_data.max(), float(np.percentile(M, 99))) * 1.05
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 14)

    cmap = plt.cm.viridis
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cf = ax.contourf(T, S, M, levels=levels, cmap=cmap, norm=norm)
    cs = ax.contour(T, S, M, levels=levels[::2], colors="#e8f4fc", linewidths=0.55, alpha=0.45)
    ax.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"{v:.2g} kg", colors="#e8f4fc")

    markers = {
        "frameless": "o",
        "flat": "s",
        "inrunner": "^",
        "outrunner": "D",
        "industrial": "P",
        "integrated": "X",
        "hub": "*",
    }
    legend_handles = []
    for form in FORM_COLORS:
        subset = train[train["Form"] == form]
        if subset.empty:
            continue
        marker = markers.get(form, "o")
        ax.scatter(
            subset["Max_Torque_Nm"],
            subset["Max_Speed_rpm"],
            c=subset["Weight_kg"],
            cmap=cmap,
            norm=norm,
            s=80,
            marker=marker,
            edgecolors="white",
            linewidths=0.75,
            zorder=5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="#4a90b8",
                markeredgecolor="white",
                markersize=8,
                linestyle="None",
                label=f"{form} ({len(subset)})",
            )
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Max / peak torque (Nm)")
    ax.set_ylabel("Max speed (rpm)")
    ax.set_title("BLDC mass model — contours = predicted kg; dots = measured")
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(s_lo, s_hi)

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Mass (kg)", color="#e8f4fc")
    cbar.ax.yaxis.set_tick_params(color="#6b8ba4")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#6b8ba4")
    cbar.outline.set_edgecolor("#2a4060")

    leg = ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        framealpha=0.92,
        facecolor="#121f36",
        edgecolor="#2a4060",
        labelcolor="#e8f4fc",
        title="Form (marker)",
        title_fontsize=8,
    )
    leg.get_title().set_color("#6b8ba4")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BLDC motor mass model")
    parser.add_argument("--torque", type=float, help="Peak torque (Nm) for a one-shot estimate")
    parser.add_argument("--speed", type=float, help="Max speed (rpm) for a one-shot estimate")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_motor_mass_fit.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--out-contour",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "robot_motor_mass_contour.png"
        ),
        help="Output torque–speed mass contour path",
    )
    args = parser.parse_args()

    df = load_motors()
    train = train_frame(df)
    fit = fit_mass_model(train)

    print(f"Motors in CSV: {len(df)}")
    print(f"Fit rows (listed mass + τ + ω): {fit['n']}")
    print(f"Model: log(m) = {fit['a']:.4f} + {fit['b_torque']:.4f}·log(τ) + {fit['c_speed']:.4f}·log(ω)")
    print(f"In-sample R²(log)={fit['r2']:.3f}  LOO R²={fit['r2_loo']:.3f}")
    print(f"MAE={fit['mae_kg']:.3f} kg  LOO MAE={fit['mae_loo_kg']:.3f} kg")
    print(
        "Implied scaling: mass ∝ τ^{:.2f} · ω^{:.2f}".format(fit["b_torque"], fit["c_speed"])
    )

    plot_fit(train, fit, args.out)
    plot_contour(train, fit, args.out_contour)

    if args.torque is not None and args.speed is not None:
        m = estimate_mass(fit["model"], args.torque, args.speed)
        print(f"Estimate: τ={args.torque} Nm, ω={args.speed} rpm → mass ≈ {m:.3f} kg")


if __name__ == "__main__":
    main()
