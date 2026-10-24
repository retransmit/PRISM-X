"""Test suite for External Backbone Replay Mode (Slice 26).

Enforces the 12 formal invariants and acceptance criteria defined in:
spec/20_backbone_replay.yaml and spec/19_detector_contract.yaml.
"""

from pathlib import Path
import sys
import tempfile
from typing import Dict

import pytest
import numpy as np

from prism_generator.backbone_replay import (
    BackboneReplayEngine,
    compute_convexity_injection_count,
    read_ulb_arff_stream,
)
from prism_generator.contract import load_contract
from prism_generator.stream import request_payload


def test_convexity_injection_count_formula():
    """Validates the mathematical convexity formula: K = round((p * N) / (1 - p))."""
    # At p = 0.02 and N = 10,000: K = round(200 / 0.98) = 204
    k_02 = compute_convexity_injection_count(10000, 0.02)
    assert k_02 == 204
    assert k_02 / (10000 + k_02) == pytest.approx(0.02, abs=1e-4)

    # At p = 0.05 and N = 10,000: K = round(500 / 0.95) = 526
    k_05 = compute_convexity_injection_count(10000, 0.05)
    assert k_05 == 526
    assert k_05 / (10000 + k_05) == pytest.approx(0.05, abs=1e-4)

    # At p = 0.0: K = 0
    assert compute_convexity_injection_count(10000, 0.0) == 0


def test_clean_backbone_purges_corpus_fraud_rows(fixture_arff):
    """Invariant 4: Clean backbone mode purges corpus-native fraud rows."""
    # Read first 1000 records without clean_backbone vs with clean_backbone
    clean_stream = list(read_ulb_arff_stream(fixture_arff, clean_backbone=True, max_records=500))
    assert len(clean_stream) == 500
    assert all(not r.is_original_fraud for r in clean_stream)


def test_every_replayed_amount_is_the_corpus_amount_verbatim(fixture_arff):
    """Invariant 1: Every replayed legitimate amount equals the corpus amount verbatim."""
    raw_stream = list(read_ulb_arff_stream(fixture_arff, clean_backbone=True, max_records=200))
    raw_amounts = [r.amount for r in raw_stream]

    engine = BackboneReplayEngine(
        corpus_path=fixture_arff,
        corpus_type="ulb",
        fraud_prevalence=0.0,  # Zero attacks
        clean_backbone=True,
        seed=42,
    )

    replayed = list(engine.replay_stream(max_legitimate_records=200))
    replayed_amounts = [r["amount"] for r in replayed]

    assert len(replayed_amounts) == len(raw_amounts)
    # Must match bitwise identically
    np.testing.assert_array_equal(replayed_amounts, raw_amounts)


def test_every_replayed_time_equals_corpus_time_plus_anchor(fixture_arff):
    """Invariant 2: Every replayed timestamp equals corpus time + anchor."""
    anchor = 1704067200.0  # 2024-01-01 00:00:00 UTC
    raw_stream = list(read_ulb_arff_stream(fixture_arff, clean_backbone=True, max_records=200))
    expected_times = [anchor + r.tx_time_seconds for r in raw_stream]

    engine = BackboneReplayEngine(
        corpus_path=fixture_arff,
        corpus_type="ulb",
        time_anchor_seconds=anchor,
        fraud_prevalence=0.0,
        clean_backbone=True,
        seed=42,
    )

    replayed = list(engine.replay_stream(max_legitimate_records=200))
    actual_times = [r["tx_time_seconds"] for r in replayed]

    np.testing.assert_array_equal(actual_times, expected_times)


def test_authorisation_request_has_zero_label_leakage(fixture_arff):
    """Invariant 3: Authorisation request feeds carry zero target labels or counterfactuals."""
    engine = BackboneReplayEngine(
        corpus_path=fixture_arff,
        corpus_type="ulb",
        fraud_prevalence=0.05,
        clean_backbone=True,
        seed=42,
    )

    records = list(engine.replay_stream(max_legitimate_records=300))
    contract = load_contract()

    for r in records:
        # Request payload must contain only REQUEST_FIELDS
        req = request_payload(r)
        assert "is_fraud" not in req
        assert "counterfactual_twin" not in req
        assert "analytical_shapley_probability" not in req
        assert "response_code" not in req
        assert "auth_code" not in req

        # Check spec/19 schema violations
        violations = contract.violations("authorisation_request", req)
        assert not violations, f"Contract violations: {violations}"


def test_fraud_prevalence_matches_requested_share_under_convexity_correction(fixture_arff):
    """Invariant 6: Realized fraud prevalence matches requested share within statistical bound."""
    target_p = 0.04
    engine = BackboneReplayEngine(
        corpus_path=fixture_arff,
        corpus_type="ulb",
        fraud_prevalence=target_p,
        clean_backbone=True,
        seed=101,
    )

    records = list(engine.replay_stream(max_legitimate_records=1000))
    n_fraud = sum(1 for r in records if r["is_fraud"] == 1)
    realized_p = n_fraud / len(records)

    # With 1000 legit rows, expected K = round(40 / 0.96) = 42, total = 1042, share = 42/1042 = 0.0403
    assert realized_p == pytest.approx(target_p, abs=0.005)


def test_backbone_replay_ledger_balance_conserved(fixture_arff):
    """Invariant 8: World ledger conservation discrepancy is strictly below 1e-6."""
    engine = BackboneReplayEngine(
        corpus_path=fixture_arff,
        corpus_type="ulb",
        fraud_prevalence=0.03,
        clean_backbone=True,
        seed=42,
    )

    _ = list(engine.replay_stream(max_legitimate_records=400))
    conserved, discrepancy = engine.double_entry.verify_global_balance_conservation()
    assert conserved is True
    assert discrepancy == pytest.approx(0.0, abs=1e-6)


def test_pca_columns_completely_excluded_from_export(fixture_arff):
    """Invariant 9: Anonymized PCA columns (V1-V28) never appear in any record."""
    engine = BackboneReplayEngine(
        corpus_path=fixture_arff,
        corpus_type="ulb",
        fraud_prevalence=0.02,
        clean_backbone=True,
        seed=42,
    )

    records = list(engine.replay_stream(max_legitimate_records=100))
    for r in records:
        for i in range(1, 29):
            assert f"V{i}" not in r, f"PCA component V{i} leaked into record!"


def test_export_dataset_writes_parquet_views(fixture_arff):
    """Integration: export_dataset creates auth_stream and gateway_telemetry parquet views."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = BackboneReplayEngine(
            corpus_path=fixture_arff,
            corpus_type="ulb",
            fraud_prevalence=0.03,
            clean_backbone=True,
            seed=42,
        )

        res = engine.export_dataset(output_dir=tmp_dir, max_legitimate_records=500, chunk_size=250)
        assert res["total_records"] > 500
        assert res["total_fraud"] > 0
        assert res["double_entry_discrepancy"] == pytest.approx(0.0, abs=1e-6)

        out_path = Path(tmp_dir)
        auth_stream_dir = out_path / "auth_stream"
        gw_stream_dir = out_path / "gateway_telemetry"

        assert auth_stream_dir.exists()
        assert gw_stream_dir.exists()
        assert any(auth_stream_dir.glob("part_*.parquet"))
        assert any(gw_stream_dir.glob("part_*.parquet"))


# =================================================================================================
# Slice 26 statistical falsification specification (statistical auditor)
#
# Three claims are tested here, each with an exact falsification test and a power control that
# proves the test can actually fail. Every constant is one of three things: read from a spec
# file, measured on the corpus, or derived from integer rounding. Nothing is a chosen tolerance.
#
# 1. ZERO LEAKAGE. No label, no counterfactual and no provenance key may reach a request
#    payload or an export view, and no exported column may identify the injected attacks above
#    the limits spec/07_export_leakage_gate.yaml already sets. The identifier columns spec/07
#    excludes (transaction_id and friends) get their own rule, because that is exactly where a
#    replay engine can hide a tell: a format that only fraud rows carry is a label, whatever the
#    gate's column exclusions say.
#
# 2. CLOCK AND AMOUNTS. The merged stream is time monotonic (non-decreasing: ULB carries 159,836
#    zero-length inter-arrivals, so strict monotonicity would be false on the real corpus), the
#    backbone subsequence keeps the corpus clock up to the declared anchor, the corpus window is
#    neither shortened nor stretched, and the amount distribution of the replayed backbone is the
#    corpus amount distribution exactly: sorted arrays equal and Wasserstein-1 distance 0.0 on
#    the grader's own log10(max(amount, 0.01) + 1) transform (benchmark_reporter.py lines 984 and
#    998). A one-cent perturbation of one row moves that distance off zero, which is asserted as
#    the power control.
#
# 3. PACING. The delivered prevalence is K / (N + K) where K = round(p * N / (1 - p))
#    (spec/20_backbone_replay.yaml pacing_and_purity.convexity_correction), so the delivered share
#    is bounded by the integer rounding of K, not by a statistical tolerance: |s - p| <= 0.5/(N+K).
#    Under superimposition the pacing must also hold per block of the backbone, because spec/07
#    section 19 paces the attack stream to the realised legitimate rate. Attacks placed at a
#    constant rate in time are rejected by a 3-sigma multinomial bound computed per block, and
#    the same rejection is demonstrated as a power control.
#
# Fixtures: a synthetic ARFF in the ULB layout (31 columns: Time, V1..V28, Amount, Class) so the
# specification runs on a clean clone where corpora/ is gitignored. The real corpus is exercised
# by the skip-guarded tests at the end, which fingerprint the file against spec/20 rather than
# against a byte hash: spec/20's canonical_sha256 (fdaf12739b82...) matches neither the hash the
# external scorecard recorded (fdaf12730dc1...) nor the local rebuilt file (4fcde0f8cc38..., a
# rebuild from the Kaggle CSV as .gitignore documents), so bytes cannot identify this corpus.
# =================================================================================================

import importlib.util as _ilu
import json as _json
import re as _re
from functools import lru_cache as _lru_cache

import yaml as _yaml
from scipy import stats as _stats

from prism_generator.cli import AUTH_STREAM_COLUMNS, GATEWAY_TELEMETRY_COLUMNS
from prism_generator.spec_loader import _find_spec_dir
from prism_generator.stream import REQUEST_FIELDS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ULB_ARFF_PATH = _REPO_ROOT / "corpora" / "creditcard.arff"  # gitignored; rebuilt per .gitignore

_SPEC_07 = _yaml.safe_load((_find_spec_dir() / "07_export_leakage_gate.yaml").read_text(encoding="utf-8"))
_SPEC_19 = _yaml.safe_load((_find_spec_dir() / "19_detector_contract.yaml").read_text(encoding="utf-8"))
_SPEC_20 = _yaml.safe_load((_find_spec_dir() / "20_backbone_replay.yaml").read_text(encoding="utf-8"))
_GATE_LIMITS = _SPEC_07["export_leakage_gate"]["thresholds"]
_MIN_SUPPORT = _GATE_LIMITS["fraud_only_value_min_support"]  # 25, spec/07

# Keys that must never appear in a detector-facing payload or view. The first block is the label
# and its ground-truth companions, the second is the counterfactual and attribution ground truth
# (spec/20 field_provenance tier 4), the third is replay provenance: in this mode every backbone
# row is legitimate and every injected row is an attack, so a provenance key IS the label.
LEAK_BANNED_KEYS = frozenset(
    {
        "is_fraud", "scenario_id", "scenario_tag", "risk_score", "base_risk", "hawkes_intensity_R",
        "syndicate_id", "botnet_cluster_id", "mule_ring_id", "beneficiary_account_id",
        "ip_subnet_prefix", "device_fingerprint_id", "counterfactual_twin",
        "counterfactual_input_deltas", "counterfactual_mode", "normative_baseline",
        "analytical_shapley_probability", "analytical_shapley_log_odds", "dominant_causal_driver",
        "explanation_narrative", "origin", "is_backbone", "backbone_row_id", "corpus_row_index",
        "corpus_source", "is_original_fraud",
    }
)
# A substring scan catches a key added later under a new name.
LEAK_BANNED_SUBSTRINGS = ("counterfactual", "shapley", "backbone", "corpus", "origin", "label")

# spec/19 fields whose value legitimately moves when an attack spends on the same card: a real
# fraud debits a real balance, so the issuer's later authorisations on that card see it. They are
# excluded from the injection-invariance comparison and from nothing else.
STATE_DEPENDENT_FIELDS = frozenset(
    {"available_balance", "credit_limit", "response_code", "auth_response_code", "auth_code",
     "tx_count_1h", "tx_count_24h"}
)

# Fixture geometry, all measured or derived (see the block above).
N_BACKBONE = 6000                  # clean rows in the fixture, after the 12 Class==1 rows are purged
N_FIXTURE_FRAUD_ROWS = 12          # 0.20 percent, against ULB's measured 0.1727 percent
FIXTURE_SPAN_SECONDS = 172792.0    # ULB's measured span (spec/20 corpora.ulb_creditcard.time_span_seconds)
FIXTURE_BUSY_SECONDS = 25920.0     # first 3,000 rows land in this first window (7.2 hours)
FIXTURE_CLOCK_HEAD_ROWS = 600      # one row per second, giving the fixture ULB's 1.0 s clock resolution
FIXTURE_BLOCKS = 10                # equal-count pacing blocks of 600 backbone rows
PREVALENCE_SWEEP = (0.02, 0.04, 0.05)


def _injection_count(n_legit: int, p: float) -> int:
    """K = round(p * N / (1 - p)) expressed as floor(x + 0.5), the integer rounding the bound below
    is derived from. It equals compute_convexity_injection_count() for every parameter here, which
    the pacing test asserts."""
    return int(np.floor(p * n_legit / (1.0 - p) + 0.5))


def _log_amounts(amounts) -> np.ndarray:
    """The certification grader's own amount transform (benchmark_reporter.py lines 984, 998)."""
    return np.log10(np.maximum(np.asarray(amounts, dtype=np.float64), 0.01) + 1.0)


def _log_seconds_deltas(times) -> np.ndarray:
    """log10(1 + dt) over strictly positive inter-arrivals, the grader's inter-arrival transform
    (benchmark_reporter.py lines 1006, 1015). Zero-length intervals are dropped exactly as the
    grader drops them, which reports/pillar1_external_corpus_scorecard.md finding 3 requires:
    ULB carries 159,836 of them and the clock resolution must be reported next to the number."""
    deltas = np.diff(np.asarray(times, dtype=np.float64))
    deltas = deltas[deltas > 0]
    return np.log10(deltas + 1.0)


@_lru_cache(maxsize=1)
def _gate_module():
    """The spec/07 gate's own code, loaded from its test file so the arithmetic here and the
    arithmetic of the export leakage gate cannot drift apart (the principle
    reports/pillar1_external_corpus_scorecard.md states for the external scorer)."""
    path = _REPO_ROOT / "tests" / "test_synthesizer" / "test_export_leakage_gate.py"
    spec = _ilu.spec_from_file_location("_slice26_gate_helpers", path)
    module = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@_lru_cache(maxsize=1)
def _arrival_resolution_fn():
    """scripts/score_external_corpus.py::_arrival_resolution, the function that prints dt_min and
    dt_median with every inter-arrival score. Reused so the clock-resolution figures this file
    asserts are computed by the same code that produced the published scorecard."""
    path = _REPO_ROOT / "scripts" / "score_external_corpus.py"
    spec = _ilu.spec_from_file_location("_slice26_score_script", path)
    module = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._arrival_resolution


def _write_ulb_layout_arff(path: Path, times, amounts, classes) -> None:
    """Writes a corpus in the ULB ARFF layout read_ulb_arff_stream parses: Time, V1..V28, Amount,
    Class, 31 comma-separated values per data line. The PCA columns are written as 0.0 because the
    loader requires 31 columns and spec/20 excludes V1..V28 from every export schema."""
    lines = ["@relation slice26_backbone_fixture"]
    lines.append("@attribute Time numeric")
    lines.extend(f"@attribute V{i} numeric" for i in range(1, 29))
    lines.append("@attribute Amount numeric")
    lines.append("@attribute Class numeric")
    lines.append("@data")
    for t, a, c in zip(times, amounts, classes):
        lines.append(",".join([f"{float(t):.1f}", *["0.0"] * 28, f"{float(a):.2f}", str(int(c))]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_fixture_arrays():
    """Deterministic fixture corpus.

    Amounts: lognormal calibrated on ULB's published moments, median 22.00 and mean 88.35 EUR
    (reports/pillar1_external_corpus_scorecard.md corpus detail), rounded to whole cents as every
    one of ULB's 284,315 legitimate amounts is (measured), with row 100 at exactly 0.00 because
    ULB carries 1,798 zero-amount legitimate rows (measured).
    Times: integer seconds like ULB's clock. The first FIXTURE_CLOCK_HEAD_ROWS rows are one second
    apart, so the fixture's own clock resolution is 1.0 s exactly as spec/20 records for ULB
    (clock_resolution_seconds, and the scorecard's measured d_min). The rest of the first
    FIXTURE_BUSY_SECONDS window holds the remaining rows up to 3,000 in that window, and 3,012 rows
    fill the remaining span: a 5.64x density contrast against ULB's measured 4.38x contrast between
    its busiest and quietest 6-hour window. The contrast is what gives the per-block pacing test
    the power to reject a constant-rate placer. Every 50th timestamp is copied from its
    predecessor, giving 120 deliberate same-second ties against ULB's measured 159,836.
    """
    rng = np.random.default_rng(2610)
    n_rows = N_BACKBONE + N_FIXTURE_FRAUD_ROWS
    mu = np.log(22.0)
    sigma = float(np.sqrt(2.0 * np.log(88.35 / 22.0)))
    amounts = np.round(rng.lognormal(mu, sigma, n_rows), 2)
    amounts[100] = 0.00

    n_busy = 3000                      # rows in the first FIXTURE_BUSY_SECONDS window, as above
    head = np.arange(FIXTURE_CLOCK_HEAD_ROWS, dtype=np.float64)          # 0 .. 599, one per second
    busy = np.floor(
        np.linspace(FIXTURE_CLOCK_HEAD_ROWS, FIXTURE_BUSY_SECONDS, n_busy - FIXTURE_CLOCK_HEAD_ROWS, endpoint=False)
    )
    quiet = np.floor(np.linspace(FIXTURE_BUSY_SECONDS, FIXTURE_SPAN_SECONDS, n_rows - n_busy, endpoint=True))
    times = np.concatenate([head, busy, quiet])
    tie_index = np.arange(50, len(times), 50)
    times[tie_index] = times[tie_index - 1]

    classes = np.zeros(n_rows, dtype=int)
    # 37 + 500k for k in 0..11 gives exactly 12 rows inside a 6,012-row corpus, the last at 92
    # percent of the span, so attacks are present in every pacing block.
    classes[np.arange(37, n_rows, 500)] = 1
    assert int(classes.sum()) == N_FIXTURE_FRAUD_ROWS
    assert len(times) == len(amounts) == len(classes) == n_rows, "fixture arrays must be the same length"
    return times, amounts, classes


@pytest.fixture(scope="module")
def fixture_arff(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("backbone_fixture") / "fixture_creditcard.arff"
    _write_ulb_layout_arff(path, *_build_fixture_arrays())
    return path


@pytest.fixture(scope="module")
def fixture_corpus(fixture_arff):
    """What the engine's own loader reads back: 6,000 clean rows, 6,012 rows when corpus fraud is
    kept. Comparisons below are made against this, not against the arrays in memory, so the test
    measures what the engine actually consumed."""
    clean = list(read_ulb_arff_stream(fixture_arff, clean_backbone=True))
    all_rows = list(read_ulb_arff_stream(fixture_arff, clean_backbone=False))
    return {"path": fixture_arff, "clean": clean, "all": all_rows}


def _replay(corpus_path: Path, fraud_prevalence: float, n_legit: int, seed: int = 42):
    engine = BackboneReplayEngine(
        corpus_path=corpus_path,
        corpus_type="ulb",
        fraud_prevalence=fraud_prevalence,
        clean_backbone=True,
        seed=seed,
    )
    return engine, list(engine.replay_stream(max_legitimate_records=n_legit))


@pytest.fixture(scope="module")
def control_stream(fixture_corpus):
    """Injection-free replay: the backbone alone, p = 0."""
    return _replay(fixture_corpus["path"], 0.0, N_BACKBONE)


@pytest.fixture(scope="module")
def superimposed_stream(fixture_corpus):
    """The same backbone with p = 0.05 superimposed: K = round(0.05 * 6000 / 0.95) = 316 attacks."""
    return _replay(fixture_corpus["path"], 0.05, N_BACKBONE)


def _backbone_rows(records):
    return [r for r in records if r.get("is_fraud") == 0]


def _attack_rows(records):
    return [r for r in records if r.get("is_fraud") == 1]


@pytest.fixture(scope="module")
def gate_rows(tmp_path_factory, superimposed_stream) -> list:
    """The two export views written and read back exactly as the spec/07 gate does it, joined on
    transaction_id, so the leakage statistics below are computed on what a detector trains from."""
    _engine, records = superimposed_stream
    out_dir = tmp_path_factory.mktemp("slice26_views")
    return _gate_module().export_and_join(records, out_dir)


# -------------------------------------------------------------------------------------------------
# Invariant group 1: zero leakage in the request payload and in the detector-facing views
# -------------------------------------------------------------------------------------------------

def test_no_request_payload_carries_a_label_a_counterfactual_or_a_provenance_key(superimposed_stream):
    """The payload claim, stated so it cannot be satisfied by accident.

    request_payload() filters on REQUEST_FIELDS today, so the structural part of this test guards
    the allowlist itself (spec/19 must never admit these keys) and any future payload builder that
    does not filter. The anti-vacuity assertion at the end requires the counterfactual and
    attribution ground truth to exist on the engine records: dropping it would also silence the
    leak, and the causal pillar would lose its ground truth."""
    _engine, records = superimposed_stream
    fraud = _attack_rows(records)
    legit = _backbone_rows(records)
    assert fraud and legit, "the fixture must produce both classes or the test proves nothing"

    for record in records:
        payload = request_payload(record)
        keys = set(payload)
        leaked = keys & LEAK_BANNED_KEYS
        assert not leaked, f"label, counterfactual or provenance key in the request payload: {sorted(leaked)}"
        for key in keys:
            flagged = [s for s in LEAK_BANNED_SUBSTRINGS if s in key.lower()]
            assert not flagged, f"request payload key {key!r} matches banned substrings {flagged}"
        _json.dumps(payload)  # a payload that cannot be serialised cannot be posted either

    for key in LEAK_BANNED_KEYS:
        assert key not in REQUEST_FIELDS, f"spec/19 authorisation_request admits {key!r}"
    for key in REQUEST_FIELDS:
        flagged = [s for s in LEAK_BANNED_SUBSTRINGS if s in key.lower()]
        assert not flagged, f"spec/19 authorisation_request field {key!r} matches {flagged}"

    # The ground truth must still exist where the causal and supervision pillars read it.
    with_counterfactual = [r for r in records if "counterfactual_twin" in r]
    with_attribution = [r for r in records if "analytical_shapley_probability" in r]
    assert with_counterfactual and with_attribution, "counterfactual ground truth was dropped"
    assert all(r.get("is_fraud") == 1 for r in with_counterfactual + with_attribution)
    assert not [r for r in legit if "counterfactual_twin" in r], "a legitimate row carries a counterfactual"


def test_the_detector_facing_views_carry_no_label_provenance_or_counterfactual_column(superimposed_stream):
    """The same claim on the columns that are actually written to disk. The column lists come from
    spec/19, so this fails when the contract grows a leaking column, which the payload-level test
    would never see (the CSV writer drops unknown keys with extrasaction="ignore")."""
    _engine, records = superimposed_stream
    assert set(AUTH_STREAM_COLUMNS) | set(GATEWAY_TELEMETRY_COLUMNS) <= set(REQUEST_FIELDS) | {
        "response_code", "auth_response_code", "auth_code"
    }, "the export views are the request feed plus the authorisation outcome (spec/19)"
    for column in tuple(AUTH_STREAM_COLUMNS) + tuple(GATEWAY_TELEMETRY_COLUMNS):
        assert column not in LEAK_BANNED_KEYS, f"export view carries {column!r}"
        flagged = [s for s in LEAK_BANNED_SUBSTRINGS if s in column.lower()]
        assert not flagged, f"export view column {column!r} matches {flagged}"


def test_no_identifier_format_encodes_the_label(superimposed_stream, gate_rows):
    """The tell spec/07's gate is blind to by design.

    spec/07 excludes transaction_id, card_id, client_ip, device_canvas_hash, timestamp and
    currency from its single-column scoring, so an identifier whose FORMAT differs between classes
    passes the gate untouched. Rule: strip every digit run from each string request column and
    require that no format is seen only on fraud rows at or above spec/07's own minimum support
    of 25 rows. A format that only attacks carry is the label in a column a detector receives."""
    _engine, records = superimposed_stream
    label_by_tx = {r["transaction_id"]: int(r["is_fraud"]) for r in records}
    labels = np.array([label_by_tx[r["transaction_id"]] for r in gate_rows], dtype=int)
    assert labels.sum() >= _MIN_SUPPORT * 2, "too few fraud rows to judge identifier formats"

    fraud_formats = {}
    legit_formats = {}
    for row, label in zip(gate_rows, labels):
        for column, value in row.items():
            if column in _SPEC_07["export_leakage_gate"]["identifier_columns_excluded"]:
                # transaction_id is excluded from the gate's scoring but NOT from this rule; only
                # the columns that carry no value information at all are skipped.
                if column not in ("transaction_id", "card_id", "stan", "rrn"):
                    continue
            if not isinstance(value, str) or value == "":
                continue
            fmt = _re.sub(r"\d+", "#", value)
            (fraud_formats if label else legit_formats).setdefault(column, {}).setdefault(fmt, 0)
            (fraud_formats if label else legit_formats)[column][fmt] += 1

    leaks = []
    for column, fmts in fraud_formats.items():
        legit = legit_formats.get(column, {})
        for fmt, count in fmts.items():
            if count >= _MIN_SUPPORT and legit.get(fmt, 0) == 0:
                leaks.append((column, fmt, count))
    assert not leaks, f"an identifier format appears only on fraud rows: {leaks}"

    # Power control: the rule must fire on a fabricated tell of the same shape.
    fabricated = {f"tx_bb_atk_{i:08d}" for i in range(30)}
    fabricated_fmt = {_re.sub(r"\d+", "#", v) for v in fabricated}
    legit_ids = {f"tx_bb_{i:08d}" for i in range(600)}
    legit_fmt = {_re.sub(r"\d+", "#", v) for v in legit_ids}
    assert fabricated_fmt.isdisjoint(legit_fmt)
    assert len(fabricated) >= _MIN_SUPPORT


# The joined view drops the label (correctly), so it is recovered from the engine records once.
_LABEL_BY_TX: Dict[str, int] = {}


def test_no_exported_column_identifies_the_injected_attacks_beyond_the_gate(superimposed_stream, gate_rows):
    """The statistical claim, graded by spec/07's own functions and limits.

    Same arithmetic as tests/test_synthesizer/test_export_leakage_gate.py: time-ordered split,
    categorical codes fitted on the training window, direction-free single-column ROC-AUC, and the
    gradient-boosting PR-AUC band. Both bounds are spec/07's, not chosen here. A failure is a
    finding, not a reason to widen a limit: above the ceiling the export tells the detector which
    rows are attacks, below the floor the fraud carries no usable signal."""
    _engine, records = superimposed_stream
    gate = _gate_module()

    labels = np.array([int(r["is_fraud"]) for r in records], dtype=int)
    assert labels.sum() >= 30, f"only {labels.sum()} fraud rows, the gate needs 30"
    _LABEL_BY_TX.clear()
    _LABEL_BY_TX.update({r["transaction_id"]: int(r["is_fraud"]) for r in records})

    joined_labels = np.array([_LABEL_BY_TX[r["transaction_id"]] for r in gate_rows], dtype=int)
    assert len(joined_labels) == len(records)
    assert int(joined_labels.sum()) >= 30

    leaks = gate.fraud_only_values(gate_rows, joined_labels)
    assert not leaks, f"values never carried by a legitimate row: {leaks}"

    X, is_categorical, names, y, cut = gate.build_matrix(gate_rows, joined_labels)
    assert int(y[cut:].sum()) >= 10, "too few fraud rows in the test window to judge the gate"

    ranked = gate.single_column_scores(X, is_categorical, names, y, cut)
    top = [(name, round(auc, 3)) for auc, name in ranked[:6]]
    assert ranked[0][0] <= _GATE_LIMITS["max_single_column_roc_auc"], (
        f"a single exported column separates backbone from injected traffic: {top}"
    )

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score

    clf = HistGradientBoostingClassifier(
        categorical_features=is_categorical, **gate.EVAL["learner_params"]
    )
    clf.fit(X[:cut], y[:cut])
    pr_auc = float(average_precision_score(y[cut:], clf.predict_proba(X[cut:])[:, 1]))
    assert pr_auc <= _GATE_LIMITS["model_pr_auc_ceiling"], f"PR-AUC {pr_auc:.4f} is above the ceiling"
    assert pr_auc >= _GATE_LIMITS["model_pr_auc_floor"], f"PR-AUC {pr_auc:.4f} is below the floor"


def test_the_leakage_gate_has_power_against_a_provenance_column(gate_rows):
    """Power control for the gate above: hand the gate the column a replay engine is most likely to
    leak (which side of the stream the row came from) and it must score 1.0, well past the 0.95
    limit. If this control ever failed, a green leakage gate would mean nothing."""
    gate = _gate_module()
    rows = [dict(r, origin=("injected" if _LABEL_BY_TX[r["transaction_id"]] else "backbone")) for r in gate_rows]
    labels = np.array([_LABEL_BY_TX[r["transaction_id"]] for r in rows], dtype=int)
    X, is_categorical, names, y, cut = gate.build_matrix(rows, labels)
    ranked = dict((name, auc) for auc, name in gate.single_column_scores(X, is_categorical, names, y, cut))
    assert ranked["origin"] == pytest.approx(1.0, abs=1e-9)
    assert ranked["origin"] > _GATE_LIMITS["max_single_column_roc_auc"]


# -------------------------------------------------------------------------------------------------
# Invariant group 2: time monotonicity and exact preservation of the corpus amount distribution
# -------------------------------------------------------------------------------------------------

def _assert_amounts_preserved_exactly(corpus_amounts, backbone_amounts):
    """Exact preservation: identical multisets and identical order, then Wasserstein-1 of 0.0 on
    the grader's transform. Any tolerance here would let a re-rounded, rescaled or re-ordered
    backbone through, which is the one thing this mode exists to prevent."""
    corpus_amounts = np.asarray(corpus_amounts, dtype=np.float64)
    backbone_amounts = np.asarray(backbone_amounts, dtype=np.float64)
    assert len(backbone_amounts) == len(corpus_amounts), (
        f"{len(backbone_amounts)} replayed against {len(corpus_amounts)} corpus rows"
    )
    np.testing.assert_array_equal(backbone_amounts, corpus_amounts, err_msg="backbone order changed")
    np.testing.assert_array_equal(
        np.sort(backbone_amounts), np.sort(corpus_amounts), err_msg="amount multiset changed"
    )
    distance = float(_stats.wasserstein_distance(_log_amounts(backbone_amounts), _log_amounts(corpus_amounts)))
    assert distance == 0.0, f"Wasserstein-1 against the original corpus amounts is {distance!r}, not 0.0"


def test_the_backbone_amount_distribution_is_preserved_bit_for_bit_under_superimposition(
    fixture_corpus, superimposed_stream
):
    """Invariant 2 (amounts), under attack superimposition rather than in an injection-free run.

    The corpus half of the comparison is what read_ulb_arff_stream returned, so this measures the
    full path: file, loader, replay, merge with 316 attacks, extraction of the legitimate rows."""
    _engine, records = superimposed_stream
    corpus_amounts = [r.amount for r in fixture_corpus["clean"]]
    backbone_amounts = [r["amount"] for r in _backbone_rows(records)]
    assert len(records) == N_BACKBONE + _injection_count(N_BACKBONE, 0.05), "superimposition changed the row count"
    _assert_amounts_preserved_exactly(corpus_amounts, backbone_amounts)


def test_wasserstein_zero_is_falsifiable(control_stream, fixture_corpus):
    """Power control for the assertion above: on a copy of the corpus with one amount moved by one
    cent, the same helper must raise, and on a copy with one row dropped it must raise too. A
    check that cannot fail is not a check (reports/pillar1_external_corpus_scorecard.md makes the
    same point about its reference control)."""
    _engine, records = control_stream
    corpus_amounts = np.array([r.amount for r in fixture_corpus["clean"]], dtype=np.float64)

    perturbed = corpus_amounts.copy()
    perturbed[7] += 0.01
    with pytest.raises(AssertionError):
        _assert_amounts_preserved_exactly(corpus_amounts, perturbed)

    with pytest.raises(AssertionError):
        _assert_amounts_preserved_exactly(corpus_amounts, corpus_amounts[:-1])


def test_the_merged_stream_is_time_monotonic_and_the_corpus_window_is_untouched(
    fixture_corpus, superimposed_stream
):
    """Invariant 2 (clock): monotonicity, the corpus clock up to the declared anchor, and attacks
    that land inside the corpus window instead of stretching it.

    Non-decreasing, not strictly increasing: ULB carries 159,836 zero-length inter-arrivals
    (measured) and a strict rule would be false on the real corpus. The window assertions are what
    keep a replay from quietly re-timing the corpus or padding it, which would move the pillar 1
    inter-arrival score for reasons that have nothing to do with arrivals."""
    engine, records = superimposed_stream
    times = np.array([r["tx_time_seconds"] for r in records], dtype=np.float64)
    assert np.all(np.diff(times) >= 0.0), (
        f"merged stream runs backwards at index {int(np.argmin(np.diff(times)))}"
    )

    corpus_times = np.array([r.tx_time_seconds for r in fixture_corpus["clean"]], dtype=np.float64)
    backbone_times = np.array([r["tx_time_seconds"] for r in _backbone_rows(records)], dtype=np.float64)
    anchor = float(engine.time_anchor_seconds)
    np.testing.assert_array_equal(
        backbone_times, anchor + corpus_times, err_msg="the replayed backbone left the corpus clock"
    )

    attack_times = np.array([r["tx_time_seconds"] for r in _attack_rows(records)], dtype=np.float64)
    window_start = anchor + corpus_times[0]
    window_end = anchor + corpus_times[-1]
    assert np.all(attack_times >= window_start) and np.all(attack_times <= window_end), (
        "an injected attack landed outside the corpus window"
    )
    assert times.min() == window_start and times.max() == window_end, (
        f"the replayed window is [{times.min()}, {times.max()}], the corpus window is "
        f"[{window_start}, {window_end}]"
    )


def test_the_replayed_inter_arrivals_keep_the_corpus_clock_resolution(fixture_corpus, superimposed_stream):
    """The arrival side of invariant 2, in the units the published scorecard reports: dt_min and
    dt_median next to the distance, because reports/pillar1_external_corpus_scorecard.md finding 3
    shows an inter-arrival score is unreadable without the clock it came from. Re-anchoring to the
    epoch must not move a single interval: for integer-second corpora (ULB and BankSim both are)
    (a + anchor) - (b + anchor) equals a - b exactly."""
    _engine, records = superimposed_stream
    corpus_times = [r.tx_time_seconds for r in fixture_corpus["clean"]]
    backbone_times = [r["tx_time_seconds"] for r in _backbone_rows(records)]

    np.testing.assert_array_equal(np.diff(backbone_times), np.diff(corpus_times))
    distance = float(_stats.wasserstein_distance(_log_seconds_deltas(backbone_times), _log_seconds_deltas(corpus_times)))
    assert distance == 0.0, f"inter-arrival Wasserstein-1 against the corpus is {distance!r}, not 0.0"

    resolve = _arrival_resolution_fn()
    corpus_min, corpus_median = resolve(corpus_times)
    replay_min, replay_median = resolve(backbone_times)
    assert (replay_min, replay_median) == (corpus_min, corpus_median)
    assert replay_min == _SPEC_20["corpora"]["ulb_creditcard"]["clock_resolution_seconds"]


def test_injection_preserves_legitimate_transactions_identically(control_stream, superimposed_stream):
    """spec/20 INV_INJECTION_INVARIANCE: injecting attacks must not change the replayed legitimate
    population.

    Compared field by field between an injection-free replay and the same corpus with 316 attacks
    superimposed. The excluded fields are the ones an attack physically moves: a fraud spends on a
    real card, so that card's balance and the issuer's later ISO 8583 answer on it change, and no
    honest implementation could keep them fixed. Everything else is a property of the backbone row
    alone: its corpus values, its identity, its merchant routing and the attributes drawn for it.
    Holding that requires the backbone to be generated from its own RNG stream and its own row
    index, so that the attack count cannot shift it."""
    _control_engine, control = control_stream
    _attack_engine, superimposed = superimposed_stream
    control_backbone = _backbone_rows(control)
    superimposed_backbone = _backbone_rows(superimposed)
    assert len(control_backbone) == len(superimposed_backbone) == N_BACKBONE

    fields = [
        f for f in REQUEST_FIELDS
        if f not in STATE_DEPENDENT_FIELDS and f in control_backbone[0]
    ]
    assert len(fields) >= 20, f"only {len(fields)} request fields compared, the list is too narrow"

    mismatches = []
    for index, (before, after) in enumerate(zip(control_backbone, superimposed_backbone)):
        for field in fields:
            if before[field] != after[field]:
                mismatches.append((index, field, before[field], after[field]))
                if len(mismatches) >= 5:
                    break
        if len(mismatches) >= 5:
            break
    assert not mismatches, (
        "superimposition changed the legitimate rows (field, index, without attacks, with attacks): "
        f"{mismatches}"
    )


def test_replayed_amounts_keep_the_corpus_currency_the_spec_declares(fixture_corpus, superimposed_stream):
    """Units, not decoration: spec/20 declares ULB nominal_currency EUR and spec/19's currency
    domain admits EUR, INR and USD, so admitting the corpus's own currency is available to the
    engine. Amounts are preserved verbatim by INV_REPLAY_AMOUNT_VERBATIM, so relabelling them in
    another currency is not: the only honest options are to emit EUR for this corpus or to fail.
    A replay that stamps EUR amounts USD is a unit error a detector trains on."""
    _engine, records = superimposed_stream
    declared = _SPEC_20["corpora"]["ulb_creditcard"]["nominal_currency"]
    domain = {
        f.name: f.domain
        for f in load_contract().fields("authorisation_request")
    }["currency"]
    seen = {r["currency"] for r in _backbone_rows(records)}
    assert declared in domain, (
        f"spec/19 currency domain {list(domain)} does not admit the corpus currency {declared!r}; "
        "spec/19 must be extended or spec/20 must stop declaring it, and converting the amounts "
        "is excluded because INV_REPLAY_AMOUNT_VERBATIM holds the amounts verbatim"
    )
    assert seen == {declared}, f"replayed rows carry {seen}, spec/20 declares {declared}"


def test_the_fixture_backbone_carries_the_properties_the_invariants_are_stated_over(fixture_corpus):
    """Data inspection for the fixture itself (the spec-first workflow's inspection step), so a
    later edit to the fixture cannot quietly remove the conditions these invariants assume."""
    clean = fixture_corpus["clean"]
    times = np.array([r.tx_time_seconds for r in clean], dtype=np.float64)
    amounts = np.array([r.amount for r in clean], dtype=np.float64)

    assert len(clean) == N_BACKBONE
    assert len(fixture_corpus["all"]) == N_BACKBONE + N_FIXTURE_FRAUD_ROWS, "corpus fraud not readable"
    assert np.all(times == np.floor(times)), "the fixture clock must be integer seconds like ULB"
    assert np.all(np.diff(times) >= 0.0)
    assert int((np.diff(times) == 0).sum()) >= 100, "same-second ties are needed for monotonicity"
    assert times[-1] - times[0] == FIXTURE_SPAN_SECONDS

    assert amounts.min() == 0.0, "a zero-amount row is required: ULB has 1,798 of them"
    assert np.all(np.isclose(amounts * 100.0, np.round(amounts * 100.0))), "amounts must be whole cents"
    assert len(np.unique(amounts)) < len(amounts), "repeated ticket sizes are required"

    busy = times[times < FIXTURE_BUSY_SECONDS]
    quiet = times[times >= FIXTURE_BUSY_SECONDS]
    contrast = (len(busy) / FIXTURE_BUSY_SECONDS) / (len(quiet) / (FIXTURE_SPAN_SECONDS - FIXTURE_BUSY_SECONDS))
    assert contrast >= 4.0, (
        f"density contrast {contrast:.2f}x is too flat to reject a constant-rate placer; ULB's own "
        "measured 6-hour contrast is 4.38x"
    )


# -------------------------------------------------------------------------------------------------
# Invariant group 3: attack prevalence pacing accuracy under superimposition
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("p", PREVALENCE_SWEEP)
def test_the_delivered_prevalence_matches_the_request_within_the_integer_rounding_bound(
    fixture_corpus, p
):
    """The total delivered share, with a bound derived from the rounding and nothing else.

    K = round(p*N/(1-p)) gives share s = K/(N+K); because s is 1-Lipschitz in K with derivative
    N/(N+K)^2 <= 1/(N+K), rounding K by at most half a unit moves the share by at most 0.5/(N+K).
    At N=6,000 that bound is 8e-5 at p=0.05, four orders of magnitude tighter than the 0.005
    tolerance the first draft of this file used, and it is arithmetic rather than a choice."""
    n_legit = N_BACKBONE
    expected_k = _injection_count(n_legit, p)
    assert compute_convexity_injection_count(n_legit, p) == expected_k, "K formula drifted from spec/20"

    _engine, records = _replay(fixture_corpus["path"], p, n_legit)
    n_total = len(records)
    n_fraud = len(_attack_rows(records))

    assert n_total == n_legit + expected_k, f"{n_total} rows for {n_legit} + {expected_k}"
    assert n_fraud == expected_k, f"{n_fraud} attacks injected, {expected_k} planned"
    assert expected_k >= 20, "too few attacks for the bound to mean anything"

    delivered = n_fraud / n_total
    bound = 0.5 / (n_legit + expected_k)
    assert abs(delivered - p) <= bound, (
        f"delivered {delivered:.6f} against requested {p}, bound {bound:.6e}"
    )


@pytest.mark.parametrize("p", PREVALENCE_SWEEP)
def test_the_rounding_bound_rejects_the_naive_injection_count(p):
    """Power control for the test above: the count a naive implementation would inject
    (K = p*N, which delivers p/(1+p)) must fall outside the very bound the test enforces, so a
    regression to naive counting is caught rather than absorbed. Measured at N=6,000: the naive
    error is 3.9e-4 at p=0.02 and 2.4e-3 at p=0.05, against bounds of 8.2e-5 and 7.9e-5."""
    n_legit = N_BACKBONE
    naive_k = int(np.floor(p * n_legit + 0.5))
    naive_share = naive_k / (n_legit + naive_k)
    naive_bound = 0.5 / (n_legit + naive_k)
    assert abs(naive_share - p) > naive_bound, (
        f"at p={p} the naive count is inside the bound, so the pacing test cannot distinguish "
        "the convexity formula from it; raise N_BACKBONE or drop this parameter"
    )


def test_attacks_are_paced_to_the_backbones_own_row_rate_in_every_block(fixture_corpus, superimposed_stream):
    """Pacing accuracy where it is visible: per block of backbone rows, not only in total.

    spec/07 section 19 paces the attack stream to the realised legitimate rate, so a block holding
    1/Q of the legitimate rows must hold about K/Q attacks. The bound admits both honest designs:
    a deterministic proportional allocation (integer rounding, error at most 1) and a multinomial
    draw (3 sigma of sqrt(K*f*(1-f)), here 16.0, plus 1 for rounding). It rejects a placer that
    puts attacks at a constant rate in time, which the power control below measures on this
    fixture's own block edges: 22.1 attacks of error against a bound of 17.0."""
    _engine, records = superimposed_stream
    corpus_times = np.array([r.tx_time_seconds for r in fixture_corpus["clean"]], dtype=np.float64)
    k_total = len(_attack_rows(records))
    q = FIXTURE_BLOCKS
    assert k_total == _injection_count(N_BACKBONE, 0.05)

    edges = np.append(corpus_times[np.arange(0, N_BACKBONE, N_BACKBONE // q)], corpus_times[-1])
    attack_times = np.array([r["tx_time_seconds"] - _engine.time_anchor_seconds for r in _attack_rows(records)])
    block_index = np.clip(np.searchsorted(edges, attack_times, side="right") - 1, 0, q - 1)

    expected = k_total / q
    sigma = float(np.sqrt(k_total * (1.0 / q) * (1.0 - 1.0/q)))
    bound = 3.0 * sigma + 1.0
    assert expected >= 25.0, f"only {expected:.1f} attacks per block, the 3-sigma rule needs 25+"

    counts = np.array([int((block_index == i).sum()) for i in range(q)])
    deviation = np.abs(counts - expected)
    assert int(deviation.max()) <= bound, (
        f"per-block attack counts {counts.tolist()} against {expected:.1f} expected, bound {bound:.1f}; "
        "attacks are not being paced to the backbone's own row rate"
    )

    # Power control: what a constant-rate placer would deliver on these exact block edges.
    spans = np.diff(edges)
    constant_rate = k_total * spans / (corpus_times[-1] - corpus_times[0])
    constant_error = float(np.abs(constant_rate - expected).max())
    assert constant_error > bound, (
        f"the fixture cannot distinguish a density-blind placer: its error would be "
        f"{constant_error:.1f} against a bound of {bound:.1f}"
    )


def test_the_running_prevalence_tracks_the_target_through_the_replay(fixture_corpus, superimposed_stream):
    """The same accuracy on the running share, which is what a detector trained mid-stream sees.
    Cumulative counts after each block must stay inside the multinomial 3-sigma band around
    K * f, with 1 added for integer allocation. On this fixture a constant-rate placer drifts to
    110.6 attacks of error by the end of the busy half, against a band of 27.7."""
    _engine, records = superimposed_stream
    corpus_times = np.array([r.tx_time_seconds for r in fixture_corpus["clean"]], dtype=np.float64)
    k_total = len(_attack_rows(records))
    q = FIXTURE_BLOCKS

    edges = np.append(corpus_times[np.arange(0, N_BACKBONE, N_BACKBONE // q)], corpus_times[-1])
    attack_times = np.array([r["tx_time_seconds"] - _engine.time_anchor_seconds for r in _attack_rows(records)])
    block_index = np.clip(np.searchsorted(edges, attack_times, side="right") - 1, 0, q - 1)
    counts = np.array([int((block_index == i).sum()) for i in range(q)])

    cumulative_attacks = np.cumsum(counts)
    fractions = np.arange(1, q + 1) / q
    expected = k_total * fractions
    bound = 3.0 * np.sqrt(k_total * fractions * (1.0 - fractions)) + 1.0
    error = np.abs(cumulative_attacks - expected)

    assert float(error.max()) <= float(bound.max()), (
        f"running attack counts {cumulative_attacks.tolist()} against expected "
        f"{np.round(expected, 1).tolist()}, band {np.round(bound, 1).tolist()}"
    )
    # The early blocks are the discriminating ones, so judge them individually rather than against
    # the loosest band in the run.
    for i in range(q):
        assert error[i] <= bound[i], (
            f"after block {i + 1}: {cumulative_attacks[i]} attacks against {expected[i]:.1f} "
            f"expected, band {bound[i]:.1f}"
        )


# -------------------------------------------------------------------------------------------------
# The real corpus: same invariants, measured on ULB, skipped when the gitignored file is absent
# -------------------------------------------------------------------------------------------------

_ULB_AVAILABLE = pytest.mark.skipif(
    not _ULB_ARFF_PATH.exists(),
    reason=f"ULB corpus not present at {_ULB_ARFF_PATH}; .gitignore keeps it out of the repository",
)


def _ulb_fingerprint_ok() -> bool:
    """Identifies the corpus by the figures spec/20 and the published scorecard record, because
    bytes cannot: spec/20's canonical_sha256 (fdaf12739b82...) matches neither the scorecard's
    (fdaf12730dc1...) nor this machine's rebuilt file (4fcde0f8cc38...)."""
    all_rows = list(read_ulb_arff_stream(_ULB_ARFF_PATH, clean_backbone=False))
    clean = list(read_ulb_arff_stream(_ULB_ARFF_PATH, clean_backbone=True))
    spec = _SPEC_20["corpora"]["ulb_creditcard"]
    amounts = np.array([r.amount for r in all_rows], dtype=np.float64)
    times = np.array([r.tx_time_seconds for r in all_rows], dtype=np.float64)
    d_min, _d_median = _arrival_resolution_fn()(times)
    return (
        len(all_rows) == spec["record_count_total"]
        and len(clean) == spec["record_count_legitimate"]
        and len(all_rows) - len(clean) == spec["record_count_fraud"]
        and float(np.median(amounts)) == 22.0
        and float(times.max() - times.min()) == spec["time_span_seconds"]
        and d_min == spec["clock_resolution_seconds"]
    )


def test_the_real_ulb_backbone_is_replayed_amount_for_amount(tmp_path_factory):
    """Invariant 2 against the original corpus itself: Wasserstein-1 of 0.0 on ULB's own amounts
    and ULB's own inter-arrivals, with the stream monotonic around 316 superimposed attacks.

    The first 6,000 legitimate rows are replayed, which is 6,734 seconds of ULB (measured): the
    slice is small enough for a unit test and large enough for the pacing bound to bind."""
    if not _ULB_ARFF_PATH.exists():
        pytest.skip(f"ULB corpus not present at {_ULB_ARFF_PATH}")
    if not _ulb_fingerprint_ok():
        pytest.skip("the file at the corpus path is not the corpus spec/20 describes")

    n_legit = N_BACKBONE
    corpus = list(read_ulb_arff_stream(_ULB_ARFF_PATH, clean_backbone=True, max_records=n_legit))
    engine, records = _replay(_ULB_ARFF_PATH, 0.05, n_legit)

    backbone = _backbone_rows(records)
    assert len(backbone) == n_legit == len(corpus)
    assert len(records) == n_legit + _injection_count(n_legit, 0.05)
    _assert_amounts_preserved_exactly([r.amount for r in corpus], [r["amount"] for r in backbone])

    times = np.array([r["tx_time_seconds"] for r in records], dtype=np.float64)
    assert np.all(np.diff(times) >= 0.0), "the real corpus stream is not time monotonic"
    anchor = float(engine.time_anchor_seconds)
    np.testing.assert_array_equal(
        [r["tx_time_seconds"] for r in backbone], anchor + np.array([r.tx_time_seconds for r in corpus])
    )
    np.testing.assert_array_equal(
        np.diff([r["tx_time_seconds"] for r in backbone]),
        np.diff([r.tx_time_seconds for r in corpus]),
    )
    distance = float(
        _stats.wasserstein_distance(
            _log_amounts([r["amount"] for r in backbone]), _log_amounts([r.amount for r in corpus])
        )
    )
    assert distance == 0.0, f"ULB amounts replayed at Wasserstein-1 {distance!r} against the corpus"


def test_the_real_ulb_prevalence_pacing_holds_under_superimposition():
    """Invariant 3 on the real corpus: the delivered share inside the same rounding bound."""
    if not _ULB_ARFF_PATH.exists():
        pytest.skip(f"ULB corpus not present at {_ULB_ARFF_PATH}")
    if not _ulb_fingerprint_ok():
        pytest.skip("the file at the corpus path is not the corpus spec/20 describes")

    n_legit = N_BACKBONE
    p = 0.05
    _engine, records = _replay(_ULB_ARFF_PATH, p, n_legit)
    expected_k = _injection_count(n_legit, p)
    n_fraud = len(_attack_rows(records))
    assert len(records) == n_legit + expected_k
    assert n_fraud == expected_k
    delivered = n_fraud / len(records)
    assert abs(delivered - p) <= 0.5 / (n_legit + expected_k), (
        f"delivered {delivered:.6f} against requested {p} on the real corpus"
    )
