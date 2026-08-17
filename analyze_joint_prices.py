"""
Fit a joint-module price estimator and plot the result.

Usage:
    python analyze_joint_prices.py
    python analyze_joint_prices.py --torque 20 --speed 120 --od 90 --mass 0.6 --type qdd
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from joint_price_model import (
    TYPE_ACCURACY_ARCMIN,
    coefficient_table,
    estimate_price,
    fit_models,
    load_joints,
    metrics,
    prepare_features,
)

TYPE_COLORS = {
    "qdd": "#2ecc71",
    "planetary": "#1abc9c",
    "harmonic": "#3498db",
    "cycloidal": "#9b59b6",
    "series-elastic": "#e74c3c",
    "hobby-servo": "#f39c12",
}


def _style_axes(ax):
    ax.set_facecolor("#0a1628")
    ax.grid(True, alpha=0.2, color="#2a4060")
    ax.tick_params(colors="#6b8ba4")
    for spine in ax.spines.values():
        spine.set_color("#2a4060")
    ax.xaxis.label.set_color("#e8f4fc")
    ax.yaxis.label.set_color("#e8f4fc")
    ax.title.set_color("#00d4ff")


def plot_fit(fitted, output_path: str) -> None:
    train = fitted.train.copy()
    train["pred"] = fitted.y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#0a1628")

    ax = axes[0]
    _style_axes(ax)
    for joint_type, color in TYPE_COLORS.items():
        subset = train[train["Type"] == joint_type]
        if subset.empty:
            continue
        ax.scatter(
            subset["Torque_Nm"],
            subset["Cost_USD"],
            s=np.clip(subset["Weight_kg"].fillna(0.4) * 180, 30, 280),
            c=color,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            label=f"{joint_type} ({len(subset)})",
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peak / stall torque (Nm)")
    ax.set_ylabel("List price (USD)")
    ax.set_title("Joint price vs torque")
    ax.text(
        0.98,
        0.02,
        "Circle size = mass",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#6b8ba4",
        style="italic",
    )
    ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        facecolor="#121f36",
        edgecolor="#2a4060",
        labelcolor="#e8f4fc",
    )

    ax = axes[1]
    _style_axes(ax)
    lo, hi = 30, 9000
    ax.plot([lo, hi], [lo, hi], color="#6b8ba4", linestyle="--", linewidth=1, zorder=1)
    for joint_type, color in TYPE_COLORS.items():
        subset = train[train["Type"] == joint_type]
        if subset.empty:
            continue
        ax.scatter(
            subset["Cost_USD"],
            subset["pred"],
            s=55,
            c=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual price (USD)")
    ax.set_ylabel("Fitted price (USD)")
    ax.set_title("Two-stage fit (type + torque, then extras)")
    stats = metrics(fitted)
    ax.text(
        0.03,
        0.97,
        (
            f"n = {stats['n_train']}\n"
            f"in-sample R² (log) = {stats['r2_in_sample']:.2f}\n"
            f"LOO R² (log) = {stats['r2_loo']:.2f}\n"
            f"LOO MAPE = {stats['mape_loo_pct']:.0f}%"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color="#e8f4fc",
        bbox=dict(boxstyle="round", facecolor="#121f36", edgecolor="#2a4060", alpha=0.95),
    )

    fig.suptitle(
        "Robot joint price estimator  |  log(price) ~ torque, speed, size, mass, accuracy, features",
        color="#00d4ff",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#0a1628", edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def print_report(fitted) -> None:
    stats = metrics(fitted)
    coefs = coefficient_table(fitted)
    print("=" * 72)
    print("JOINT PRICE MODEL")
    print("=" * 72)
    print(f"Training rows (listed + user quotes, no estimates): {stats['n_train']}")
    print(f"Torque elasticity (d log P / d log T): {stats['torque_elasticity']:.3f}")
    print(f"Precision elasticity (d log P / d log 1/acc): {stats['precision_elasticity']:.3f}")
    print(f"In-sample R² (log price):      {stats['r2_in_sample']:.3f}")
    print(f"Leave-one-out R² (log price):  {stats['r2_loo']:.3f}")
    print(f"In-sample MAE:                 ${stats['mae_usd']:.0f}")
    print(f"Leave-one-out MAE:             ${stats['mae_loo_usd']:.0f}")
    print(f"Leave-one-out MAPE:            {stats['mape_loo_pct']:.0f}%")
    print()
    print("Stage 1: type + log(torque) + log(1/accuracy). Stage 2: non-negative residual adjustments.")
    print("approx_x_factor is exp(log_coef); type dummies are vs the dropped baseline class.")
    print()
    print(coefs.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print()
    print("Typical accuracy used when a spec sheet has no repeatability:")
    for name, value in TYPE_ACCURACY_ARCMIN.items():
        print(f"  {name:16s}  {value:6.2f} arcmin")
    print()

    train = fitted.train.copy()
    train["pred"] = fitted.y_pred
    train["residual_pct"] = (train["pred"] - train["Cost_USD"]) / train["Cost_USD"] * 100
    print("Largest residuals (predicted vs listed):")
    show = train.assign(abs_res=train["residual_pct"].abs()).sort_values("abs_res", ascending=False)
    cols = ["Name", "MFG", "Type", "Torque_Nm", "Cost_USD", "pred", "residual_pct"]
    print(show[cols].head(8).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and query the joint price estimator")
    parser.add_argument("--torque", type=float, help="Peak torque (Nm)")
    parser.add_argument("--speed", type=float, help="Rated output speed (rpm)")
    parser.add_argument("--od", type=float, help="Outer diameter (mm)")
    parser.add_argument("--mass", type=float, default=None, help="Mass (kg); estimated if omitted")
    parser.add_argument("--accuracy", type=float, default=None, help="Output accuracy (arcmin)")
    parser.add_argument("--type", dest="joint_type", default="qdd", help="Joint class")
    parser.add_argument("--encoder-bits", type=float, default=None)
    parser.add_argument("--dual-encoder", action="store_true", default=None)
    parser.add_argument("--single-encoder", action="store_true")
    parser.add_argument("--brake", action="store_true")
    parser.add_argument("--no-driver", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_joints()
    prepared = prepare_features(raw)
    fitted = fit_models(raw)
    print_report(fitted)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, "robot_joint_price_fit.png")
    plot_fit(fitted, plot_path)
    print(f"Saved plot to: {plot_path}")

    if args.torque is not None:
        dual = True if args.dual_encoder else False if args.single_encoder else True
        speed = args.speed if args.speed is not None else float(np.nanmedian(prepared["Rated_Speed_rpm"]))
        od = args.od if args.od is not None else float(np.nanmedian(prepared["OD_mm"]))
        result = estimate_price(
            fitted,
            torque_nm=args.torque,
            speed_rpm=speed,
            od_mm=od,
            mass_kg=args.mass,
            accuracy_arcmin=args.accuracy,
            joint_type=args.joint_type,
            encoder_bits=args.encoder_bits,
            dual_encoder=dual,
            has_brake=args.brake,
            has_driver=not args.no_driver,
        )
        print()
        print("PRICE ESTIMATE")
        print(f"  Type:       {args.joint_type}")
        print(f"  Torque:     {args.torque:.2f} Nm")
        print(f"  Speed:      {speed:.1f} rpm")
        print(f"  OD:         {od:.1f} mm")
        print(f"  Mass:       {result['mass_kg']:.3f} kg" + (" (estimated)" if result["mass_was_estimated"] else ""))
        print(f"  Accuracy:   {result['accuracy_arcmin']:.2f} arcmin")
        print(f"  Encoder:    {result['encoder_bits']:.0f} bit" + (" dual" if dual else " single"))
        print(f"  Brake:      {'yes' if args.brake else 'no'}")
        print(f"  Driver:     {'yes' if not args.no_driver else 'no'}")
        print(f"  Est. price: ${result['cost_usd']:.0f}")


if __name__ == "__main__":
    main()
