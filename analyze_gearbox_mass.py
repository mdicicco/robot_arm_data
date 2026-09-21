"""
Fit gearbox mass / size models by style.

Within a given frame, mass is nearly independent of ratio (harmonic, worm,
cycloidal). Across the catalog, mass tracks torque capacity and type far more
than ratio alone:

    log(mass) ≈ a[type] + b · log(T_out) + c · log(ratio)

Usage:
    python analyze_gearbox_mass.py
    python analyze_gearbox_mass.py --ratio 50 --torque-out 40 --type harmonic
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TYPE_COLORS = {
    "harmonic": "#3498db",
    "cycloidal": "#9b59b6",
    "planetary": "#2ecc71",
    "worm": "#e67e22",
    "spur": "#e74c3c",
}


def data_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gearbox_data.csv")


def load_gearboxes(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or data_path())


def train_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["Weight_kg", "Ratio", "Rated_Output_Torque_Nm", "Type"]).copy()
    out = out[(out["Weight_kg"] > 0) & (out["Ratio"] > 0) & (out["Rated_Output_Torque_Nm"] > 0)]
    out["log_mass"] = np.log(out["Weight_kg"].astype(float))
    out["log_ratio"] = np.log(out["Ratio"].astype(float))
    out["log_torque"] = np.log(out["Rated_Output_Torque_Nm"].astype(float))
    out["log_od"] = np.log(out["OD_mm"].astype(float).replace(0, np.nan))
    return out


def fit_global(train: pd.DataFrame) -> dict:
    """log(m) = a[type] + b·log(T) + c·log(ratio)"""
    X = train[["Type", "log_torque", "log_ratio"]]
    y = train["log_mass"].to_numpy()
    pre = ColumnTransformer(
        [
            ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
            ("num", "passthrough", ["log_torque", "log_ratio"]),
        ]
    )
    pipe = Pipeline([("pre", pre), ("lr", LinearRegression())])
    pipe.fit(X, y)
    y_hat = pipe.predict(X)
    y_loo = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    # recover numeric coefs after one-hot
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
        "c_ratio": float(coef.get("num__log_ratio", np.nan)),
        "intercept": float(lr.intercept_),
        "coef": coef,
        "n": len(train),
        "y_hat": y_hat,
    }


def fit_per_type(train: pd.DataFrame) -> pd.DataFrame:
    """Per-type: log(m)=a+b·log(T) and log(m)=a+c·log(ratio)."""
    rows = []
    for typ, g in train.groupby("Type"):
        if len(g) < 5:
            continue
        for xcol, label in [("log_torque", "torque"), ("log_ratio", "ratio")]:
            X = g[[xcol]].to_numpy()
            y = g["log_mass"].to_numpy()
            m = LinearRegression().fit(X, y)
            y_hat = m.predict(X)
            rows.append(
                {
                    "Type": typ,
                    "vs": label,
                    "n": len(g),
                    "a": float(m.intercept_),
                    "slope": float(m.coef_[0]),
                    "r2": float(r2_score(y, y_hat)),
                    "mae_kg": float(mean_absolute_error(np.exp(y), np.exp(y_hat))),
                }
            )
    return pd.DataFrame(rows)


def estimate_mass(pipe, typ: str, torque_out: float, ratio: float) -> float:
    X = pd.DataFrame(
        [{"Type": typ, "log_torque": np.log(torque_out), "log_ratio": np.log(ratio)}]
    )
    return float(np.exp(pipe.predict(X)[0]))


def _style(ax):
    ax.set_facecolor("#0a1628")
    ax.grid(True, alpha=0.2, color="#2a4060")
    ax.tick_params(colors="#6b8ba4")
    for spine in ax.spines.values():
        spine.set_color("#2a4060")
    ax.xaxis.label.set_color("#e8f4fc")
    ax.yaxis.label.set_color("#e8f4fc")
    ax.title.set_color("#00d4ff")


def plot_overview(train: pd.DataFrame, per_type: pd.DataFrame, fit: dict, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#0a1628")

    # Left: mass vs ratio
    ax = axes[0]
    _style(ax)
    for typ, color in TYPE_COLORS.items():
        g = train[train["Type"] == typ]
        if g.empty:
            continue
        ax.scatter(
            g["Ratio"],
            g["Weight_kg"],
            s=np.clip(g["Rated_Output_Torque_Nm"].astype(float) * 1.5, 25, 220),
            c=color,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            label=f"{typ} ({len(g)})",
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gear ratio")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Mass vs ratio (size ∝ output torque)")
    ax.text(
        0.98,
        0.02,
        "Circle size = rated output torque",
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

    # Right: mass vs output torque — the real driver
    ax = axes[1]
    _style(ax)
    markers = {"harmonic": "o", "cycloidal": "s", "planetary": "D", "worm": "^", "spur": "P"}
    for typ, color in TYPE_COLORS.items():
        g = train[train["Type"] == typ]
        if g.empty:
            continue
        ax.scatter(
            g["Rated_Output_Torque_Nm"],
            g["Weight_kg"],
            s=55,
            c=color,
            marker=markers.get(typ, "o"),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            label=typ,
            zorder=3,
        )
        # per-type torque fit line
        row = per_type[(per_type["Type"] == typ) & (per_type["vs"] == "torque")]
        if not row.empty and row.iloc[0]["r2"] > 0.3:
            a, b = row.iloc[0]["a"], row.iloc[0]["slope"]
            t = np.logspace(
                np.log10(g["Rated_Output_Torque_Nm"].min()),
                np.log10(g["Rated_Output_Torque_Nm"].max()),
                40,
            )
            ax.plot(t, np.exp(a + b * np.log(t)), color=color, linewidth=1.2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rated output torque (Nm)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title(
        f"Mass vs torque by type  |  global R²={fit['r2']:.2f}  "
        f"(b_T={fit['b_torque']:.2f}, c_i={fit['c_ratio']:.2f})"
    )
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        facecolor="#121f36",
        edgecolor="#2a4060",
        labelcolor="#e8f4fc",
    )

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


def plot_od(train: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("#0a1628")
    _style(ax)
    for typ, color in TYPE_COLORS.items():
        g = train[train["Type"] == typ].dropna(subset=["OD_mm"])
        if g.empty:
            continue
        ax.scatter(
            g["Rated_Output_Torque_Nm"],
            g["OD_mm"],
            s=np.clip(g["Ratio"].astype(float) * 1.2, 20, 180),
            c=color,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            label=f"{typ} ({len(g)})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rated output torque (Nm)")
    ax.set_ylabel("OD / housing diameter (mm)")
    ax.set_title("Gearbox size vs torque (marker size ∝ ratio)")
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        facecolor="#121f36",
        edgecolor="#2a4060",
        labelcolor="#e8f4fc",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Gearbox mass / size model")
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--torque-out", type=float, dest="torque_out")
    parser.add_argument("--type", default="harmonic", choices=list(TYPE_COLORS))
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_gearbox_mass_fit.png"),
    )
    parser.add_argument(
        "--out-od",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_gearbox_od_fit.png"),
    )
    args = parser.parse_args()

    df = load_gearboxes()
    train = train_frame(df)
    fit = fit_global(train)
    per_type = fit_per_type(train)

    print(f"Gearboxes in CSV: {len(df)}")
    print(f"Fit rows: {fit['n']}")
    print(f"Types: {train['Type'].value_counts().to_dict()}")
    print(
        f"Global: log(m)=a[type]+{fit['b_torque']:.3f}·log(T_out)+{fit['c_ratio']:.3f}·log(ratio)"
    )
    print(
        f"  R²(log)={fit['r2']:.3f}  LOO R²={fit['r2_loo']:.3f}  "
        f"MAE={fit['mae_kg']:.3f} kg  LOO MAE={fit['mae_loo_kg']:.3f} kg"
    )
    print("\nPer-type slopes (mass ∝ x^slope):")
    for _, r in per_type.sort_values(["Type", "vs"]).iterrows():
        print(
            f"  {r['Type']:10s} vs {r['vs']:6s}: slope={r['slope']:+.3f}  "
            f"R²={r['r2']:.2f}  MAE={r['mae_kg']:.2f} kg  n={r['n']}"
        )

    plot_overview(train, per_type, fit, args.out)
    plot_od(train, args.out_od)

    if args.ratio is not None and args.torque_out is not None:
        m = estimate_mass(fit["pipe"], args.type, args.torque_out, args.ratio)
        print(
            f"\nEstimate: {args.type}  i={args.ratio}  T_out={args.torque_out} Nm "
            f"→ mass ≈ {m:.3f} kg"
        )


if __name__ == "__main__":
    main()
