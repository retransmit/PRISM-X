"""Tests for scripts/report_grade_stability.py, which is where the sweep numbers come from.

Every count quoted in reports/certification_grade_stability.md is produced by this
script rather than transcribed, so the assertions here check the two properties the
report depends on: one row per stored run, and gate grouping that ignores the observed
values (two runs failing one gate must count twice, not twice-differently).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report_grade_stability as rgs  # noqa: E402


def _write_run(directory: Path, name: str, *, n: int, seed: int, violations: list) -> Path:
    path = directory / name / "benchmark_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "seed": seed,
                    "run_protocol": {"n_transactions": n},
                    "n_transactions": n,
                },
                "certification_grade": "NON_CERTIFIED_FAIL",
                "violations": violations,
                "streaming": {"n_days_evaluated": 5, "mean_pr_auc": 0.1273},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_reports_every_run_and_groups_gates_across_runs(tmp_path: Path, capsys):
    solid = "Pillar 1 (Data Fidelity): inter-arrival Wasserstein 0.4246 exceeds the 0.1500 maximum"
    other = "Pillar 1 (Data Fidelity): inter-arrival Wasserstein 0.1883 exceeds the 0.1500 maximum"
    flipping = "Pillar 3 (Operational Streaming): prequential PR-AUC 0.0417 falls below the 0.1500 minimum"

    paths = [
        _write_run(tmp_path, "a", n=500, seed=42, violations=[solid, flipping]),
        _write_run(tmp_path, "b", n=500, seed=0, violations=[other]),
        _write_run(tmp_path, "c", n=2000, seed=1, violations=[]),
    ]

    assert rgs.main([str(p) for p in paths]) == 0
    out = capsys.readouterr().out

    # one row per stored run, plus the header row
    assert out.count("| `") == 3
    assert "violation counts: 0-2 (0, 1, 2)" in out
    # the two differently-valued inter-arrival failures are one gate, failed in 2/3 runs
    assert "inter-arrival Wasserstein # exceeds the maximum | 2/3 | FLIPS" in out
    # a run with no violations must not invent a gate
    assert "failed in | reading |" in out


def test_missing_file_is_an_error_not_an_empty_answer(tmp_path: Path, capsys):
    assert rgs.main([str(tmp_path / "nope.json")]) == 2
    assert "not a file" in capsys.readouterr().err
