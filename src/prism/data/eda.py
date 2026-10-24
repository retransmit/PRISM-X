"""Module 1: exploratory analysis artifacts (Phase 1).

    python -m prism.data.eda

Returns every statistic the write-up quotes and writes every figure printed
beside it, so prose and plots cannot drift apart. Anything cited in the report
should come from here rather than being read off a notebook cell that has since
been re-run with different data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prism import config, tracks  # noqa: E402
from prism.data.loader import describe_balance, load_raw  # noqa: E402

EDA_FIGURES: tuple[str, ...] = (
    "class_balance",
    "amount_distribution",
    "temporal_pattern",
)


def output_paths() -> tuple[Path, Path]:
    """Where the EDA numbers and figures go: the ULB track's own directories,
    beside the metrics the results generator reads."""
    return tracks.ULB.metrics_dir / "eda.json", tracks.ULB.figures_dir


def generate_eda_report(
    df: pd.DataFrame | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    """Write the Phase 1 figures and return the numbers that go in the prose."""
    frame = load_raw() if df is None else df
    target = tracks.ULB.target_column

    figures_dir = Path(out_dir) if out_dir else output_paths()[1]
    figures_dir.mkdir(parents=True, exist_ok=True)

    fraud = frame[frame[target] == 1]
    normal = frame[frame[target] == 0]

    _plot_class_balance(len(normal), len(fraud), figures_dir)
    _plot_amount_distribution(fraud, normal, figures_dir)
    _plot_temporal_pattern(fraud, normal, figures_dir)

    report: dict[str, Any] = dict(describe_balance(frame, target))
    report.update(
        {
            "median_amount_fraud": float(fraud["Amount"].median()),
            "median_amount_normal": float(normal["Amount"].median()),
            "mean_amount_fraud": float(fraud["Amount"].mean()),
            "mean_amount_normal": float(normal["Amount"].mean()),
            "max_amount_fraud": float(fraud["Amount"].max()),
            "figures": [str(figures_dir / f"{name}.png") for name in EDA_FIGURES],
        }
    )
    return report


def _plot_class_balance(n_normal: int, n_fraud: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["normal", "fraud"], [n_normal, n_fraud], color=["#4C78A8", "#E45756"])
    # A linear scale renders the fraud bar as literally invisible. That is the
    # point being made, but an unreadable chart makes it badly.
    ax.set_yscale("log")
    ax.set_ylabel("transactions (log scale)")
    ax.set_title(f"Class balance: fraud is {n_fraud / (n_normal + n_fraud):.3%}")
    for index, value in enumerate([n_normal, n_fraud]):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)
    plt.close(fig)


def amount_bins(normal_amounts: pd.Series, n: int = 60) -> np.ndarray:
    """Bin edges over the range the amount figure shows: zero to the 99th
    percentile of legitimate amounts. Bins over the full range (a maximum in
    the tens of thousands) put the whole legitimate distribution into one bar
    of the clipped figure, which made legitimate spending look uniform."""
    upper = float(normal_amounts.quantile(0.99))
    return np.linspace(0.0, upper, n + 1)


def _plot_amount_distribution(
    fraud: pd.DataFrame, normal: pd.DataFrame, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    edges = amount_bins(normal["Amount"])
    ax.hist(normal["Amount"], bins=edges, alpha=0.6, density=True, label="normal")
    ax.hist(fraud["Amount"], bins=edges, alpha=0.6, density=True, label="fraud")
    # The long right tail is clipped so the bulk of the distribution stays legible.
    ax.set_xlim(0, float(edges[-1]))
    ax.set_xlabel("Amount")
    ax.set_ylabel("density")
    ax.set_title("Transaction amount by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "amount_distribution.png", dpi=150)
    plt.close(fig)


def _plot_temporal_pattern(
    fraud: pd.DataFrame, normal: pd.DataFrame, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal["Time"] / 3600.0, bins=48, alpha=0.6, density=True, label="normal")
    ax.hist(fraud["Time"] / 3600.0, bins=48, alpha=0.6, density=True, label="fraud")
    ax.set_xlabel("hours since first transaction")
    ax.set_ylabel("density")
    ax.set_title("Temporal distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "temporal_pattern.png", dpi=150)
    plt.close(fig)


def main() -> None:
    report = generate_eda_report()

    metrics_path, figures_dir = output_paths()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"rows       : {report['n_total']:,}")
    print(f"fraud      : {report['n_fraud']:,} ({report['fraud_rate']:.5%})")
    print(f"imbalance  : 1 : {report['imbalance_ratio']:.1f}")
    print(
        f"median amt : fraud {report['median_amount_fraud']:.2f} "
        f"vs normal {report['median_amount_normal']:.2f}"
    )
    print(f"figures -> {figures_dir}")
    print(f"stats   -> {metrics_path}")


if __name__ == "__main__":
    main()
