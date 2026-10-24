"""Module 3: the comparison table and figures that go in the report.

    python -m prism.evaluation.benchmark
    python -m prism.evaluation.benchmark --track india

Rebuilds the same splits as training (identical seed => identical partition),
loads every persisted model of the track, and evaluates all of them on the
untouched test set at their tuned thresholds. On the Indian track it also
draws the precision-recall curve re-weighted to one fraud in 80,847 (design
D2) and adds the track's extra columns to the table (design D6).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt  # noqa: E402

from prism import tracks  # noqa: E402
from prism.data.frames import load_frame  # noqa: E402
from prism.data.preprocess import prepare_splits  # noqa: E402
from prism.evaluation.metrics import EvalResult, adjusted_curve, evaluate, track_extras  # noqa: E402
from prism.evaluation.threshold import downsample_curve, pr_curve_points  # noqa: E402
from prism.training import manifest  # noqa: E402
from prism.models import load_all  # noqa: E402


def figures_dir():
    return tracks.active().figures_dir


def metrics_dir():
    return tracks.active().metrics_dir


def load_thresholds() -> dict[str, float]:
    path = tracks.active().model_dir / "thresholds.json"
    if not path.exists():
        raise SystemExit(f"No tuned thresholds at {path}; run `python -m prism.training.train` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(results: list[EvalResult], extras: dict[str, dict] | None = None) -> str:
    extras = extras or {}
    columns = [
        "Model", "Threshold", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1",
        "Missed fraud", "Alerts per catch",
    ]
    if extras:
        columns += ["Precision at 1 in 80,847", "FPR per 10k", "Value above floor caught"]
    header = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
    rows = []
    for r in sorted(results, key=lambda r: r.pr_auc, reverse=True):
        cells = [
            r.model, f"{r.threshold:.4f}", f"**{r.pr_auc:.4f}**", f"{r.roc_auc:.4f}", f"{r.precision:.4f}",
            f"{r.recall:.4f}", f"{r.f1:.4f}", str(r.false_negatives),
            f"{r.alerts_per_true_fraud:.1f}" if r.true_positives else "n/a",
        ]
        if extras:
            e = extras.get(r.model, {})
            cells += [
                f"{e.get('adjusted_precision_at_prevalence', 0.0):.4f}",
                f"{e.get('fpr_per_10k', 0.0):.1f}",
                f"{e.get('share_above_floor_caught', 0.0):.3f}",
            ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, *rows])


def plot_pr_curves(curves: dict[str, dict], baseline: float, path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, curve in curves.items():
        ax.plot(curve["recall"], curve["precision"], label=name, linewidth=1.8)
    # The no-skill line for PR is the positive prevalence, not 0.5.
    ax.axhline(
        baseline, linestyle="--", linewidth=1, color="grey",
        label=f"no-skill ({baseline:.6f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results: list[EvalResult]) -> None:
    figures_dir().mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4))
    axes = [axes] if len(results) == 1 else list(axes)

    for ax, r in zip(axes, results):
        matrix = [[r.true_negatives, r.false_positives],
                  [r.false_negatives, r.true_positives]]
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{r.model}\nPR-AUC {r.pr_auc:.3f}")
        ax.set_xticks([0, 1], ["pred normal", "pred fraud"])
        ax.set_yticks([0, 1], ["actual normal", "actual fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir() / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the persisted models on the test split.")
    tracks.add_track_argument(parser)
    schema = tracks.activate_from_args(parser.parse_args())

    models = load_all(schema.model_dir)
    if not models:
        raise SystemExit(
            f"No trained models in {schema.model_dir}. "
            f"Run `python -m prism.training.train --track {schema.name}` first."
        )

    frame = load_frame(schema)
    splits = prepare_splits(frame, schema=schema)
    thresholds = load_thresholds()
    prevalence = schema.setting("prevalence_target", None)
    amounts = frame.loc[splits.X_test.index, schema.amount_column].to_numpy(dtype=float)

    results: list[EvalResult] = []
    curves: dict[str, dict] = {}
    adjusted: dict[str, dict] = {}
    extras: dict[str, dict] = {}

    for name, model in models.items():
        scores = model.risk_score(splits.X_test)
        if name not in thresholds:
            raise SystemExit(f"No tuned threshold for {name}; retrain it before benchmarking.")
        threshold = thresholds[name]
        result = evaluate(splits.y_test, scores, threshold, model=name)
        results.append(result)
        curves[name] = pr_curve_points(splits.y_test, scores)
        if prevalence is not None:
            adjusted[name] = adjusted_curve(splits.y_test, scores, float(prevalence))
            extras[name] = track_extras(
                splits.y_test, scores, threshold, amounts,
                prevalence=float(prevalence), floor=float(schema.setting("value_floor")),
                prevalence_range=schema.setting("prevalence_range", None),
            )
        print(result)

    figures = figures_dir()
    plot_pr_curves(
        curves, float(splits.y_test.mean()), figures / "pr_curves.png",
        "Precision-Recall curves (test window)",
    )
    if adjusted:
        plot_pr_curves(
            adjusted, float(prevalence), figures / "pr_curves_adjusted.png",
            f"Precision-Recall at one fraud in {1 / float(prevalence):,.0f}",
        )
    plot_confusion_matrices(results)

    out = metrics_dir()
    out.mkdir(parents=True, exist_ok=True)
    applied = {
        n: bool(e.get("resample_applied", False))
        for n, e in (manifest.load(schema.model_dir) or {}).get("models", {}).items()
    }
    (out / "benchmark.json").write_text(
        json.dumps(
            [{**r.as_dict(), **extras.get(r.model, {}), "resample_applied": applied.get(r.model, False)} for r in results],
            indent=2,
        ),
        encoding="utf-8",
    )
    table = markdown_table(results, extras or None)
    (out / "benchmark.md").write_text(table, encoding="utf-8")
    (out / "pr_curves.json").write_text(
        json.dumps({name: downsample_curve(curve) for name, curve in curves.items()}, indent=2),
        encoding="utf-8",
    )
    if adjusted:
        (out / "adjusted_curves.json").write_text(
            json.dumps({name: downsample_curve(curve) for name, curve in adjusted.items()}, indent=2),
            encoding="utf-8",
        )

    print("\n" + table)
    print(f"\nFigures -> {figures}")
    print(f"Tables  -> {out}")


if __name__ == "__main__":
    main()
