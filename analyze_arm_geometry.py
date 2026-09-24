"""
Empirical link-length heuristic for serial arms, as fractions of reach.

Each arm in data/arm_geometry_data.csv is reduced to the lever chain seen by
the pitch joints when the arm is stretched out horizontally:

    shoulder offset → upper arm → forearm → wrist (wrist pitch → flange)

Fractions are normalized so offset + upper + forearm + wrist = 1 (the chain
length ≈ reach). Base height (floor → shoulder pitch axis) is reported as a
fraction of the same chain length; it never acts as a gravity lever arm.

Usage:
    python analyze_arm_geometry.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEGMENTS = ["offset", "upper", "forearm", "wrist"]
SEGMENT_COLUMNS = {
    "offset": "Shoulder_Offset_m",
    "upper": "Upper_Arm_m",
    "forearm": "Forearm_m",
    "wrist": "Wrist_m",
}
SEGMENT_COLORS = {
    "offset": "#6b8ba4",
    "upper": "#3498db",
    "forearm": "#2ecc71",
    "wrist": "#e67e22",
}
TYPE_COLORS = {
    "collaborative": "#2ecc71",
    "industrial": "#e74c3c",
    "research": "#9b59b6",
    "hobby": "#3498db",
}
# Arm families the GUI can base the heuristic on.
GEOMETRY_BASES = {
    "all": None,
    "collaborative": ["collaborative"],
    "industrial": ["industrial"],
    "research/hobby": ["research", "hobby"],
}


@dataclass(frozen=True)
class GeometryFractions:
    """Link lengths as fractions of reach (offset+upper+forearm+wrist = 1)."""

    base: float
    offset: float
    upper: float
    forearm: float
    wrist: float
    n: int = 0

    def lengths(self, reach_m: float) -> dict[str, float]:
        return {
            "base": self.base * reach_m,
            "offset": self.offset * reach_m,
            "upper": self.upper * reach_m,
            "forearm": self.forearm * reach_m,
            "wrist": self.wrist * reach_m,
        }

    def normalized(self) -> "GeometryFractions":
        total = self.offset + self.upper + self.forearm + self.wrist
        return GeometryFractions(
            base=self.base / total,
            offset=self.offset / total,
            upper=self.upper / total,
            forearm=self.forearm / total,
            wrist=self.wrist / total,
            n=self.n,
        )


def data_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "arm_geometry_data.csv")


def load_geometry(path: str | None = None) -> pd.DataFrame:
    """Load the geometry table and add normalized fraction columns (frac_*)."""
    df = pd.read_csv(path or data_path())
    df = df.dropna(subset=["Upper_Arm_m", "Forearm_m", "Wrist_m", "Base_Height_m"]).copy()
    df["Shoulder_Offset_m"] = df["Shoulder_Offset_m"].fillna(0.0)
    chain = sum(df[c] for c in SEGMENT_COLUMNS.values())
    df["Chain_m"] = chain
    df["chain_over_reach"] = chain / df["Reach_m"]
    for seg, col in SEGMENT_COLUMNS.items():
        df[f"frac_{seg}"] = df[col] / chain
    df["frac_base"] = df["Base_Height_m"] / chain
    return df


def fit_fractions(df: pd.DataFrame, types: list[str] | None = None) -> GeometryFractions:
    """Median fractions over the selected arm types, renormalized to sum to 1."""
    sub = df if types is None else df[df["Type"].isin(types)]
    if sub.empty:
        sub = df
    med = sub[[f"frac_{s}" for s in ["base", *SEGMENTS]]].median()
    return GeometryFractions(
        base=float(med["frac_base"]),
        offset=float(med["frac_offset"]),
        upper=float(med["frac_upper"]),
        forearm=float(med["frac_forearm"]),
        wrist=float(med["frac_wrist"]),
        n=len(sub),
    ).normalized()


def fraction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median / IQR of each fraction for every geometry basis."""
    rows = []
    for basis, types in GEOMETRY_BASES.items():
        sub = df if types is None else df[df["Type"].isin(types)]
        fit = fit_fractions(df, types)
        row = {"basis": basis, "n": len(sub)}
        for seg in ["base", *SEGMENTS]:
            q1, q3 = sub[f"frac_{seg}"].quantile([0.25, 0.75])
            row[seg] = getattr(fit, seg)
            row[f"{seg}_iqr"] = (float(q1), float(q3))
        row["chain/reach"] = float(sub["chain_over_reach"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor("#0a1628")
    ax.grid(True, alpha=0.2, color="#2a4060")
    ax.tick_params(colors="#6b8ba4")
    for spine in ax.spines.values():
        spine.set_color("#2a4060")
    ax.xaxis.label.set_color("#e8f4fc")
    ax.yaxis.label.set_color("#e8f4fc")
    ax.title.set_color("#00d4ff")


def plot_geometry(df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor("#0a1628")

    # Left: stacked fractions per arm, sorted by reach
    ax = axes[0]
    _style(ax)
    d = df.sort_values(["Type", "Reach_m"])
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for seg in SEGMENTS:
        vals = d[f"frac_{seg}"].to_numpy()
        ax.barh(y, vals, left=left, color=SEGMENT_COLORS[seg], edgecolor="#0a1628", label=seg)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{n} ({r:.2f} m)" for n, r in zip(d["Name"], d["Reach_m"])], fontsize=7
    )
    for tick, typ in zip(ax.get_yticklabels(), d["Type"]):
        tick.set_color(TYPE_COLORS.get(typ, "#e8f4fc"))
    fit = fit_fractions(df)
    edge = 0.0
    for seg in SEGMENTS:
        edge += getattr(fit, seg)
        ax.axvline(edge, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of chain length (offset + upper + forearm + wrist)")
    ax.set_title("Lever-chain split per arm (dashed = pooled median)")
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.0, -0.07), ncol=4, fontsize=8,
        facecolor="#121f36", edgecolor="#2a4060", labelcolor="#e8f4fc",
    )

    # Right: each fraction vs reach
    ax = axes[1]
    _style(ax)
    markers = {"upper": "o", "forearm": "s", "wrist": "D", "base": "^"}
    for seg, marker in markers.items():
        color = SEGMENT_COLORS.get(seg, "#f1c40f")
        ax.scatter(
            df["Reach_m"],
            df[f"frac_{seg}"],
            c=color,
            marker=marker,
            s=45,
            edgecolors="white",
            linewidths=0.4,
            label=f"{seg}: median {getattr(fit, seg):.2f}",
        )
        ax.axhline(getattr(fit, seg), color=color, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Published reach (m)")
    ax.set_ylabel("Fraction of chain length")
    ax.set_title("Fractions are ~scale-free across reach")
    ax.legend(fontsize=8, facecolor="#121f36", edgecolor="#2a4060", labelcolor="#e8f4fc")

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Arm link-length heuristic")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_arm_geometry_fit.png"),
    )
    args = parser.parse_args()

    df = load_geometry()
    print(f"Arms with geometry: {len(df)}  types={df['Type'].value_counts().to_dict()}")
    table = fraction_table(df)
    print("\nMedian fractions of chain length (IQR in brackets):")
    for _, r in table.iterrows():
        parts = [f"{seg}={r[seg]:.3f} [{r[seg + '_iqr'][0]:.2f}–{r[seg + '_iqr'][1]:.2f}]" for seg in ["base", *SEGMENTS]]
        print(f"  {r['basis']:15s} n={r['n']:2d}  chain/reach={r['chain/reach']:.2f}  " + "  ".join(parts))
    plot_geometry(df, args.out)


if __name__ == "__main__":
    main()
