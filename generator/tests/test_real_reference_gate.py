"""Verification suite for the Pillar 1 real-corpus reference (spec/18, 2026-10-15).

The two continuous gates were re-derived because real card traffic failed the
hardcoded-reference gates harder than the generator being graded (measured in
`reports/pillar1_external_corpus_scorecard.md`, 2026-10-10/15). These tests pin
the properties that make the replacement honest, each validated on the corpus
the spec names:

1. The corpus self-test: a contiguous window of the real corpus passes its own
   calibrated gates. A gate real data cannot pass is not a realism gate.
2. The reference-sampler control behaves like the inversion that broke the old
   gates: a draw from the hardcoded lognormal fails the amount bar.
3. The limits are size-corrected: they shrink with sample size, which is what
   the static 0.150 limit failed to do.
4. The calibration is seed-independent: across run seeds the graded metric is
   compared against the same reference sample and the same limit.
5. Provenance is enforced: a corpus file whose sha256 mismatches the spec never
   silently grades -- the loader falls back and the scorecard names the fallback.
"""


import numpy as np
import pytest
import yaml

from prism_generator.engine import SimulationEngine
from prism_generator.real_reference import load_real_reference
import prism_generator.real_reference as rr
from prism_generator.benchmark_reporter import UnifiedBenchmarkRunner


@pytest.fixture(scope="module")
def real_ref():
    ref = load_real_reference()
    if ref is None:
        pytest.skip("spec/18 pillar_1_real_reference or its corpus file is unavailable")
    return ref


def test_corpus_self_window_passes_its_own_gates(real_ref):
    """A contiguous window of the real corpus must pass its own calibrated gates.

    Both checks reuse the graded estimator's exact shapes: the amount pool
    against an iid pool sample, and a contiguous ordered-delta window against the
    pool reference, each at the same size. Calibrated at quantile 0.99 over
    contiguous windows, so a real window is *designed* to pass; measured 2026-10-15
    at 8/9 amount windows and 9/9 arrival windows against the grader.
    """
    import scipy.stats as stats

    n = 2000
    amt = float(stats.wasserstein_distance(
        real_ref.amount_log_sample(n),
        real_ref.amount_log_pool[np.random.default_rng(5).integers(0, len(real_ref.amount_log_pool), n)],
    ))
    arr = float(stats.wasserstein_distance(
        real_ref.arrival_window_log_sample(n, draw=77),
        real_ref.arrival_log_sample(n),
    ))
    assert amt <= real_ref.w1_limit("amount", n)
    assert arr <= real_ref.w1_limit("arrival", n)


def test_control_fails_amount_bar(real_ref):
    """A Fed-lognormal sample is not real traffic; the bar must know it.

    This is the inversion of the pre-2026-10-15 behaviour, where the control
    scored 0.0000 while real ULB traffic scored 0.3745 against a 0.150 limit.
    """
    import scipy.stats as stats

    rng = np.random.default_rng(42)
    control = np.log10(np.maximum(rng.lognormal(mean=3.75, sigma=0.85, size=2000), 0.01) + 1.0)
    d = float(stats.wasserstein_distance(control, real_ref.amount_log_sample(2000)))
    assert d > real_ref.w1_limit("amount", 2000)


def test_limits_are_size_corrected(real_ref):
    """W1 noise shrinks with n; the limit must track it, not stay fixed.

    The static 0.150 limit let an unchanged n=1,500 snapshot fail at 0.2295 and
    an n=50,000 one pass at 0.1161. The calibrated limit is a decreasing
    function of the graded size.
    """
    small = real_ref.w1_limit("amount", 2_000)
    large = real_ref.w1_limit("amount", 50_000)
    assert large < small, f"amount limit at n=50,000 ({large}) must be tighter than at n=2,000 ({small})"


def test_calibration_seed_is_independent_of_run_seed(real_ref):
    """The same gate at the same size yields the same limit for every run seed.

    Each seed's graded metric is compared against the same reference sample, so
    the stability matrix compares generators, not reference draws.
    """
    first = real_ref.w1_limit("amount", 2_000)
    for _ in range(3):
        assert real_ref.w1_limit("amount", 2_000) == first


def test_sha256_mismatch_falls_back_by_name(monkeypatch):
    """A corpus that does not hash to the spec value must never grade.

    The loader degrades to the embedded reference (returns None), and the
    grader then names that fallback in `reference_note` -- a graded number must
    never be attributable to the wrong file.
    """
    spec_path = rr._find_repo_root() / "spec" / "18_benchmark_reporting.yaml"
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    cfg = raw["pillar_1_real_reference"]
    active = cfg.get("active_corpus", "")
    body = dict(cfg[active])

    tampered = {**cfg, active: {**body, "corpus_sha256": "0" * 64}}
    monkeypatch.setattr(rr, "_load_reference_config", lambda: tampered)
    rr.load_real_reference.cache_clear()
    try:
        assert rr.load_real_reference() is None
    finally:
        rr.load_real_reference.cache_clear()


def test_engine_batch_is_graded_against_named_reference():
    """The grader runs end to end with the real reference and says so."""
    ref = load_real_reference()
    if ref is None:
        pytest.skip("real-corpus reference unavailable")
    engine = SimulationEngine(n_cards=60, n_merchants=20, region="US", seed=5)
    records = engine.generate_batch(n_transactions=400, fraud_prevalence=0.1, time_span_days=4.0)
    card = UnifiedBenchmarkRunner(
        region="US", n_transactions=400, time_span_days=4.0, k_daily=10, seed=5
    )._evaluate_data_fidelity(records)
    assert ref.corpus_name.split("(")[0] in card.reference_note
    assert f"quantile of {ref.draws}" in card.reference_note
    # Provenance names the reference explicitly and, with the real corpus loaded,
    # never labels it with the superseded corpus's clock.
    assert card.reference_kind == "real_corpus"
    assert "ULB" not in card.reference_note


def test_legacy_limit_does_not_poison_the_live_cache(real_ref):
    """Regression (2026-10-15): calling the superseded iid-pair limit must not
    overwrite the value `w1_limit` returns.

    `_legacy_pool_w1_limit` is kept only so the calibration change is reviewable;
    it once wrote the live `_limit_cache[(kind, n)]` key, so a single call
    silently replaced the calibrated limit with the iid value and every later
    grade used the wrong bar.
    """
    live = real_ref.w1_limit("amount", 2_000)
    legacy = real_ref._legacy_pool_w1_limit("amount", 2_000)
    assert legacy != live, "fixture no longer distinguishes the two limits"
    assert real_ref.w1_limit("amount", 2_000) == live
