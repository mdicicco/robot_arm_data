"""
Fit BLDC motor mass models by form (frameless / outrunner / …).

Mass tracks peak torque far more than speed. Across the catalog, form
shifts the intercept a lot (industrial/integrated are heavy for a given τ;
frameless/outrunner are light):

    log(mass) ≈ a[form] + b · log(τ_max) + c · log(ω_max)

Usage:
    python analyze_motor_mass.py
    python analyze_motor_mass.py --torque 2.0 --speed 4000 --form frameless
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FORM_COLORS = {
    "frameless": "#2ecc71",
    "flat": "#3498db",
    "inrunner": "#9b59b6",
    "outrunner": "#e67e22",
    "industrial": "#e74c3c",
    "integrated": "#1abc9c",
    "hub": "#f1c40f",
}

FORM_MARKERS = {
    "frameless": "o",
    "flat": "s",
    "inrunner": "^",
    "outrunner": "D",
    "industrial": "P",
    "integrated": "X",
    "hub": "*",
}


def data_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bldc_motor_data.csv")


def load_motors(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or data_path())


def train_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with listed mass + max torque + max speed + form."""
    out = df.dropna(subset=["Max_Torque_Nm", "Max_Speed_rpm", "Weight_kg", "Form"]).copy()
    out = out[
        (out["Weight_kg"] > 0)
        & (out["Max_Torque_Nm"] > 0)
        & (out["Max_Speed_rpm"] > 0)
    ]
    if "Weight_Flag" in out.columns:
        listed = out[out["Weight_Flag"].fillna("listed") == "listed"]
        if len(listed) >= 8:
            out = listed.copy()
    out["log_torque"] = np.log(out["Max_Torque_Nm"].astype(float))
    out["log_speed"] = np.log(out["Max_Speed_rpm"].astype(float))
    out["log_mass"] = np.log(out["Weight_kg"].astype(float))
    return out


def fit_global(train: pd.DataFrame) -> dict:
    """log(m) = a[form] + b·log(τ) + c·log(ω)"""
    X = train[["Form", "log_torque", "log_speed"]]
    y = train["log_mass"].to_numpy()
    pre = ColumnTransformer(
        [
            ("form", OneHotEncoder(handle_unknown="ignore"), ["Form"]),
            ("num", "passthrough", ["log_torque", "log_speed"]),
        ]
    )
    pipe = Pipeline([("pre", pre), ("lr", LinearRegression())])
    pipe.fit(X, y)
    y_hat = pipe.predict(X)
    y_loo = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    lr = pipe.named_steps["lr"]
    feat = pipe.named_steps["pre"].get_feature_names_out()
    coef = dict(zip(feat, lr.coef_))
    return {
        "pipe": pipe,
        "r2": float(r2_score(y, y_hat)),
        "r2_loo": float(r2_score(y, y_loo)),
        "mae_kg": float(mean_absolute_error(np.exp(y), np.exp(y_hat))),
        "mae_loo_kg": float(mean_absolute_error(np.exp(y), np.exp(y_loo))),
        "b_torque": float(coef.get("num__log_torque", np.nan)),
        "c_speed": float(coef.get("num__log_speed", np.nan)),
        "intercept": float(lr.intercept_),
        "coef": coef,
        "n": len(train),
        "y_hat": y_hat,
        "y_loo": y_loo,
    }


def fit_pooled(train: pd.DataFrame) -> dict:
    """Pooled (no form): log(m)=a+b·log(τ)+c·log(ω) — used for contour baseline."""
    X = train[["log_torque", "log_speed"]].to_numpy()
    y = train["log_mass"].to_numpy()
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    y_loo = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    return {
        "model": model,
        "a": float(model.intercept_),
        "b_torque": float(model.coef_[0]),
        "c_speed": float(model.coef_[1]),
        "r2": float(r2_score(y, y_hat)),
        "r2_loo": float(r2_score(y, y_loo)),
        "mae_kg": float(mean_absolute_error(np.exp(y), np.exp(y_hat))),
        "mae_loo_kg": float(mean_absolute_error(np.exp(y), np.exp(y_loo))),
        "n": len(train),
    }


def fit_per_form(train: pd.DataFrame) -> pd.DataFrame:
    """Per-form: log(m)=a+b·log(τ) and log(m)=a+c·log(ω)."""
    rows = []
    for form, g in train.groupby("Form"):
        if len(g) < 5:
            continue
        for xcol, label in [("log_torque", "torque"), ("log_speed", "speed")]:
            X = g[[xcol]].to_numpy()
            y = g["log_mass"].to_numpy()
            m = LinearRegression().fit(X, y)
            y_hat = m.predict(X)
            rows.append(
                {
                    "Form": form,
                    "vs": label,
                    "n": len(g),
                    "a": float(m.intercept_),
                    "slope": float(m.coef_[0]),
                    "r2": float(r2_score(y, y_hat)),
                    "mae_kg": float(mean_absolute_error(np.exp(y), np.exp(y_hat))),
                }
            )
    return pd.DataFrame(rows)


def estimate_mass(pipe, form: str, torque_nm: float, speed_rpm: float) -> float:
    X = pd.DataFrame(
        [
            {
                "Form": form,
                "log_torque": np.log(torque_nm),
                "log_speed": np.log(speed_rpm),
            }
        ]
    )
    return float(np.exp(pipe.predict(X)[0]))


def form_offsets(fit: dict, train: pd.DataFrame) -> dict[str, float]:
    """Effective intercept a_form = intercept + one-hot coef (0 for dropped baseline)."""
    forms = sorted(train["Form"].unique())
    offsets = {}
    for form in forms:
        key = f"form__Form_{form}"
        offsets[form] = fit["intercept"] + float(fit["coef"].get(key, 0.0))
    return offsets


def _style_axes(ax):
    ax.set_facecolor("#0a1628")
    ax.grid(True, alpha=0.2, color="#2a4060")
    ax.tick_params(colors="#6b8ba4")
    for spine in ax.spines.values():
        spine.set_color("#2a4060")
    ax.xaxis.label.set_color("#e8f4fc")
    ax.yaxis.label.set_color("#e8f4fc")
    ax.title.set_color("#00d4ff")


def plot_fit(
    train: pd.DataFrame,
    fit: dict,
    per_form: pd.DataFrame,
    output_path: str,
) -> None:
    train = train.copy()
    train["pred_kg"] = np.exp(fit["y_hat"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#0a1628")

    # Left: mass vs torque with per-form estimator lines
    ax = axes[0]
    _style_axes(ax)
    for form, color in FORM_COLORS.items():
        g = train[train["Form"] == form]
        if g.empty:
            continue
        ax.scatter(
            g["Max_Torque_Nm"],
            g["Weight_kg"],
            s=np.clip(8000.0 / g["Max_Speed_rpm"].astype(float), 25, 280),
            c=color,
            marker=FORM_MARKERS.get(form, "o"),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            label=f"{form} ({len(g)})",
            zorder=3,
        )
        row = per_form[(per_form["Form"] == form) & (per_form["vs"] == "torque")]
        if not row.empty and row.iloc[0]["r2"] > 0.25:
            a, b = row.iloc[0]["a"], row.iloc[0]["slope"]
            t = np.logspace(
                np.log10(g["Max_Torque_Nm"].min()),
                np.log10(g["Max_Torque_Nm"].max()),
                40,
            )
            ax.plot(t, np.exp(a + b * np.log(t)), color=color, linewidth=1.4, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Max / peak torque (Nm)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Mass vs torque by form  (lines = per-form mass∝τ^b)")
    ax.text(
        0.98,
        0.02,
        "Marker size ∝ 1/speed",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#6b8ba4",
        style="italic",
    )
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        facecolor="#121f36",
        edgecolor="#2a4060",
        labelcolor="#e8f4fc",
    )

    # Right: form-aware predicted vs actual
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
            marker=FORM_MARKERS.get(form, "o"),
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
        f"log(m)=a[form]+b·log(τ)+c·log(ω)  |  R²={fit['r2']:.2f}  LOO R²={fit['r2_loo']:.2f}"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(
        0.02,
        0.98,
        f"n={fit['n']}  MAE={fit['mae_kg']:.2f} kg  LOO MAE={fit['mae_loo_kg']:.2f} kg\n"
        f"b_τ={fit['b_torque']:.3f}  c_ω={fit['c_speed']:.3f}",
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


def plot_contour(train: pd.DataFrame, pooled: dict, fit: dict, output_path: str) -> None:
    """Torque–speed map: pooled contours for backdrop; dots = measured by form."""
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.patch.set_facecolor("#0a1628")
    _style_axes(ax)

    t_lo = train["Max_Torque_Nm"].min() * 0.7
    t_hi = train["Max_Torque_Nm"].max() * 1.15
    s_lo = max(train["Max_Speed_rpm"].min() * 0.85, 300)
    s_hi = min(train["Max_Speed_rpm"].max() * 1.05, 25000)

    tt = np.logspace(np.log10(t_lo), np.log10(t_hi), 120)
    ss = np.logspace(np.log10(s_lo), np.log10(s_hi), 120)
    T, S = np.meshgrid(tt, ss)
    M = np.exp(
        pooled["a"]
        + pooled["b_torque"] * np.log(T)
        + pooled["c_speed"] * np.log(S)
    )

    m_data = train["Weight_kg"].astype(float)
    vmin = min(m_data.min(), float(M.min())) * 0.9
    vmax = max(m_data.max(), float(np.percentile(M, 99))) * 1.05
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 14)

    cmap = plt.cm.viridis
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cf = ax.contourf(T, S, M, levels=levels, cmap=cmap, norm=norm)
    cs = ax.contour(
        T, S, M, levels=levels[::2], colors="#e8f4fc", linewidths=0.55, alpha=0.45
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"{v:.2g} kg", colors="#e8f4fc")

    legend_handles = []
    for form in FORM_COLORS:
        subset = train[train["Form"] == form]
        if subset.empty:
            continue
        marker = FORM_MARKERS.get(form, "o")
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
    ax.set_title(
        f"BLDC mass map — contours = pooled model; form-aware R²={fit['r2']:.2f}"
    )
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
    parser = argparse.ArgumentParser(description="BLDC motor mass model by form")
    parser.add_argument("--torque", type=float, help="Peak torque (Nm) for a one-shot estimate")
    parser.add_argument("--speed", type=float, help="Max speed (rpm) for a one-shot estimate")
    parser.add_argument(
        "--form",
        default="frameless",
        choices=list(FORM_COLORS),
        help="Motor form for one-shot estimate",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "robot_motor_mass_fit.png"
        ),
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
    fit = fit_global(train)
    pooled = fit_pooled(train)
    per_form = fit_per_form(train)
    offsets = form_offsets(fit, train)

    print(f"Motors in CSV: {len(df)}")
    print(f"Fit rows (listed mass + τ + ω): {fit['n']}")
    print(f"Forms: {train['Form'].value_counts().to_dict()}")
    print(
        f"Global: log(m)=a[form]+{fit['b_torque']:.3f}·log(τ)+{fit['c_speed']:.3f}·log(ω)"
    )
    print(
        f"  R²(log)={fit['r2']:.3f}  LOO R²={fit['r2_loo']:.3f}  "
        f"MAE={fit['mae_kg']:.3f} kg  LOO MAE={fit['mae_loo_kg']:.3f} kg"
    )
    print(
        f"Pooled (no form): R²={pooled['r2']:.3f}  LOO R²={pooled['r2_loo']:.3f}  "
        f"MAE={pooled['mae_kg']:.3f} kg"
    )
    print("\nEffective intercepts a[form] (log-mass units):")
    for form, a in sorted(offsets.items(), key=lambda kv: kv[1]):
        n = int((train["Form"] == form).sum())
        print(f"  {form:12s}  a={a:+.3f}  n={n}")

    print("\nPer-form slopes (mass ∝ x^slope):")
    if per_form.empty:
        print("  (need ≥5 rows per form)")
    else:
        for _, r in per_form.sort_values(["Form", "vs"]).iterrows():
            print(
                f"  {r['Form']:12s} vs {r['vs']:6s}: slope={r['slope']:+.3f}  "
                f"R²={r['r2']:.2f}  MAE={r['mae_kg']:.2f} kg  n={r['n']}"
            )

    plot_fit(train, fit, per_form, args.out)
    plot_contour(train, pooled, fit, args.out_contour)

    if args.torque is not None and args.speed is not None:
        m = estimate_mass(fit["pipe"], args.form, args.torque, args.speed)
        print(
            f"\nEstimate: {args.form}  τ={args.torque} Nm  ω={args.speed} rpm "
            f"→ mass ≈ {m:.3f} kg"
        )


if __name__ == "__main__":
    main()
