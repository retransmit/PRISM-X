"""Guards for the single causal contract declared in spec/21.

These tests exist because the eleven explained columns and the per-playbook physical
levers used to be declared three times over: in `benchmark.py`, in
`benchmark_reporter.py` and in `spec/21`. The copies had drifted. `benchmark.py`
contradicted `spec/21` for seven of the eight playbooks they shared, used a third
vocabulary (`avs_mismatch`, `cvv_match_flag`, `billing_shipping_mismatch`) and declared
six playbooks (`ADV_CARDING_MICRO_PROBE`, `COUNTERFEIT_CLONE`, `IMPOSSIBLE_TRAVEL`, ...)
that no spec defines. `spec/21` now owns both tables and the harnesses read them.
"""

import collections

import numpy as np
import yaml

from prism_generator import benchmark
from prism_generator.benchmark_reporter import UnifiedBenchmarkRunner
from prism_generator.causal_contract import (
    KNOWN_UNEMITTED_FIELDS,
    feature_names,
    feature_vector,
    ground_truth_attributions,
    ground_truth_keys,
    intervention_support,
    load_causal_contract,
    load_intervention_projection,
)
from prism_generator.contract import load_contract
from prism_generator.engine import DiscreteEventEngine
from prism_generator.spec_loader import _find_spec_dir

# Playbooks the generator emits that spec/21 declares no physical footprint for,
# measured over US and IN batches. Declaring one means deciding which record fields it
# mutates, which is a spec-authoring task grounded in spec/04 and spec/05 mechanics.
# A fourth playbook appearing here fails this test loudly instead of going unscored.
UNDECLARED_PLAYBOOKS = frozenset(
    {
        "ADV_NOCTURNAL_BURST",
        "IN_ADV_APK_SMS_STEALER",
        "IN_ADV_RENT_PORTAL_CASHOUT",
    }
)

# Levers playbooks declare that the explained model never receives, so an explainer
# cannot be credited with recovering them. Closed set: a typo in spec/21 fails.
UNMODELLED_LEVERS = frozenset(
    {
        "asn_type",
        "available_balance",
        "channel_type",
        "credit_limit",
        "eci",
        "mcc",
        "pos_entry_mode",
        "trans_status_3ds",
        "vaai_score",
    }
)


def _spec_21() -> dict:
    path = _find_spec_dir() / "21_prism_causal_evaluation.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _generated_records(region: str, seeds=(1, 42, 303)) -> list:
    records = []
    for seed in seeds:
        engine = DiscreteEventEngine(n_cards=100, n_merchants=30, region=region, seed=seed)
        records.extend(engine.generate_batch(n_transactions=1000, fraud_prevalence=0.08))
    return records


def test_the_contract_is_the_eleven_columns_spec_21_declares():
    """The columns, their sources and their ground-truth keys come from spec/21 alone."""
    declared = _spec_21()["explained_feature_contract"]["features"]
    features = load_causal_contract()

    assert len(declared) == len(features) == 11
    assert [d["name"] for d in declared] == list(feature_names())

    for entry, feature in zip(declared, features):
        assert feature.name == entry["name"]
        assert feature.kind == entry["kind"]
        assert feature.ground_truth_key == entry["ground_truth_key"]
        assert dict(feature.source) == {str(k): str(v) for k, v in entry["source"].items()}
        assert ("field" in feature.source) != ("rule" in feature.source)
        assert feature.kind in ("continuous", "binary")


def test_the_reporter_reads_the_same_contract():
    """Pillar 4 no longer keeps a second vocabulary with its own polarities."""
    runner = UnifiedBenchmarkRunner(n_transactions=100, time_span_days=2.0, seed=1)
    _, empirical, ground_truth = runner._evaluate_causal_xai([])

    assert list(empirical.keys()) == list(feature_names())
    assert list(ground_truth.keys()) == list(feature_names())


def test_intervention_support_is_exactly_the_spec_projection():
    """Every declared lever is either a modelled column or reported as unmodelled."""
    projection = load_intervention_projection()
    modelled = set(feature_names())

    assert len(projection.declared) >= 8
    for playbook, levers in projection.declared.items():
        unique = set(levers)
        support = projection.support_set(playbook)
        unmodelled = set(projection.unmodelled_levers(playbook))

        assert support == unique & modelled
        assert unmodelled == unique - modelled
        assert not support & unmodelled
        # The two halves partition the declaration: nothing is silently dropped.
        assert len(support) + len(unmodelled) == len(unique)
        assert support == intervention_support(playbook)
        assert support, f"{playbook} declares no lever the model can see"

    assert set(projection.all_unmodelled_levers()) == UNMODELLED_LEVERS


def test_every_unmodelled_lever_is_a_real_authorisation_field():
    """Catches a spec typo that would otherwise vanish into 'unmodelled'."""
    contract = load_contract()
    feed_fields = set()
    for feed in contract.feeds:
        feed_fields.update(contract.field_names(feed))

    for lever in load_intervention_projection().all_unmodelled_levers():
        assert lever in feed_fields, (
            f"spec/21 intervenes on {lever!r}, which is neither an explained column nor a "
            "field of the spec/19 authorisation contract: a typo or an undeclared derived feature"
        )


def test_every_playbook_the_generator_emits_is_declared_or_pinned():
    """A new playbook cannot enter the generator unscored and silent."""
    declared = set(load_intervention_projection().declared)
    emitted = collections.Counter()
    for region in ("US", "IN"):
        for record in _generated_records(region):
            if record.get("is_fraud") == 1:
                emitted[str(record.get("scenario_tag", ""))] += 1

    assert emitted, "no fraud records generated"
    undeclared = set(emitted) - declared
    assert undeclared <= UNDECLARED_PLAYBOOKS, (
        "these playbooks are emitted but spec/21 declares no physical footprint for them, "
        f"so the intervention metric silently skips them: {sorted(undeclared - UNDECLARED_PLAYBOOKS)}"
    )


def test_pinned_undeclared_playbooks_are_real_spec_playbooks():
    """The pinned list cannot rot into invented names, or into a duplicate declaration."""
    spec_dir = _find_spec_dir()
    spec_04 = yaml.safe_load((spec_dir / "04_adversarial_playbooks.yaml").read_text(encoding="utf-8"))
    spec_05 = yaml.safe_load((spec_dir / "05_india_payment_rails.yaml").read_text(encoding="utf-8"))
    known = {p["id"] for p in spec_04["adversarial_playbooks"]}
    known |= {p["id"] for p in spec_05["indian_adversarial_playbooks"]}

    declared = set(load_intervention_projection().declared)
    assert UNDECLARED_PLAYBOOKS <= known
    assert not UNDECLARED_PLAYBOOKS & declared


def test_every_ground_truth_key_is_emitted_by_the_generator():
    """A ground-truth key that the generator never writes would score a constant zero."""
    records = _generated_records("US", seeds=(42,))
    emitted = set()
    for record in records:
        emitted.update(record.get("analytical_shapley_probability") or {})
        emitted.update(record.get("analytical_shapley_log_odds") or {})

    for column, key in ground_truth_keys().items():
        assert key in emitted, f"column {column} is scored against {key!r}, which no record carries"


def test_the_columns_are_real_record_fields_on_every_record():
    """Every column reads a field the generator writes; no gap is left untracked.

    Until 2026-10-16 this test also carried the two known gaps (`emv_arqc_verified`,
    `three_ds_authenticated`). The rails now decide authentication before the authorisation
    step and write all five deviations onto the record, so the tracking mapping is empty;
    a future entry must be a deliberate, measured decision rather than a silent constant.
    """
    assert not KNOWN_UNEMITTED_FIELDS, (
        "KNOWN_UNEMITTED_FIELDS is no longer empty: a column the generator does not emit is a "
        "constant-column leak in every explained model. Wire it through the rails and update "
        "this test with the measurement."
    )

    records = _generated_records("US", seeds=(42,))
    present = set(records[0])

    for feature in load_causal_contract():
        field_name = feature.source.get("field")
        if field_name is None:
            continue
        assert field_name in present, f"{feature.name} reads {field_name!r}, absent from every record"


def test_the_authentication_columns_are_the_rails_outcome(monkeypatch):
    """The five authentication fields are, per record, exactly what `authenticate()` returned."""
    from prism_generator import rails as rails_module

    captured = {}
    original = rails_module.RailVerifierSwitch.authenticate

    def spy(self, intent, card):
        outcome = original(self, intent, card)
        captured[intent.tx_id] = outcome
        return outcome

    monkeypatch.setattr(rails_module.RailVerifierSwitch, "authenticate", spy)
    engine = DiscreteEventEngine(n_cards=80, n_merchants=25, region="US", seed=3)
    records = engine.generate_batch(n_transactions=600, fraud_prevalence=0.08)

    matched = [r for r in records if r.get("transaction_id") in captured]
    assert len(matched) == len(records), (
        "every emitted record must carry the authentication outcome the rails produced for it"
    )
    for record in matched:
        outcome = captured[record["transaction_id"]]
        assert record["emv_arqc_verified"] == outcome.arqc_verified
        assert record["emv_pin_verified"] == outcome.pin_verified
        assert record["emv_tvr_clean"] == outcome.tvr_clean
        assert record["three_ds_authenticated"] == outcome.three_ds_authenticated
        assert record["three_ds_attempted"] == outcome.three_ds_attempted

    # The columns carry information rather than a constant: both polarities appear in one batch.
    assert {bool(r["three_ds_authenticated"]) for r in matched} == {True, False}


def test_the_3ds_columns_track_the_recorded_acs_status():
    """On CNP rows the two 3DS flags are read off the recorded ACS status, not a second opinion."""
    records = _generated_records("IN", seeds=(42,))
    cnp_rows = [r for r in records if str(r.get("channel_type", "")).startswith("CNP")]
    assert cnp_rows, "no card-not-present rows generated"

    for record in cnp_rows:
        status = str(record.get("trans_status_3ds", ""))
        assert bool(record["three_ds_authenticated"]) == (status in ("Y", "C"))
        assert bool(record["three_ds_attempted"]) == (status in ("A", "C"))

    # Card-present rows carry no ACS status upstream; spec/19's declared domain defaults the
    # record field to "Y" while `three_ds_authenticated` stays false, so the relation is pinned
    # for CNP rows only - the rows the 3DS columns exist for.


def test_authentication_decisions_never_read_the_fraud_label():
    """The pre-authorisation decision path must not touch `is_fraud`.

    The regression this guards against: the scorer's fallback used to read the label, which made
    the five deviations a perfect oracle and let the pre-authentication risk score depend on the
    authentication result it decides. A source-level check is the strongest form available for a
    function whose behaviour on the label is exactly "never".
    """
    import inspect

    from prism_generator import causal_scm, rails as rails_module

    sources = {
        "RailVerifierSwitch.authenticate": inspect.getsource(rails_module.RailVerifierSwitch.authenticate),
        "RailVerifierSwitch._hardware_crypto_verified": inspect.getsource(
            rails_module.RailVerifierSwitch._hardware_crypto_verified
        ),
        "RailVerifierSwitch._pos_credentials": inspect.getsource(rails_module.RailVerifierSwitch._pos_credentials),
        "HeuristicBankScorer.pre_authentication_score": inspect.getsource(
            causal_scm.HeuristicBankScorer.pre_authentication_score
        ),
    }
    for name, source in sources.items():
        assert "is_fraud" not in source, f"{name} mentions the fraud label; the decision path must stay label-free"


def test_feature_vector_reads_the_record_fields_with_declared_polarity():
    """The extraction rules are the ones spec/21 names, with the scorer's polarity."""
    columns = list(feature_names())
    index = {name: i for i, name in enumerate(columns)}

    record = {
        "amount": 125.5,
        "haversine_velocity_kph": 640.0,
        "ip_distance_from_home_km": 12.0,
        "tx_count_1h": 3,
        "tx_count_24h": 7,
        "is_cross_border": True,
        "avs_match_code": "N",
        "billing_shipping_match": 0,
        "cvv_match_flag": 0,
        "emv_arqc_verified": 0,
        "three_ds_authenticated": 0,
    }
    row = feature_vector(record)

    assert row[index["amount"]] == 125.5
    assert row[index["tx_count_1h"]] == 3.0
    assert row[index["is_cross_border"]] == 1.0
    assert row[index["avs_mismatch_flag"]] == 1.0
    assert row[index["billing_shipping_mismatch"]] == 1.0
    # Mismatch polarity: the scorer's input is 1 when the value did not match, so a
    # matched CVV must read 0.0 and a mismatched CVV 1.0.
    assert row[index["cvv_mismatch_flag"]] == 1.0

    matched = dict(record, avs_match_code="Y", billing_shipping_match=1, cvv_match_flag=1, is_cross_border=None)
    matched_row = feature_vector(matched)
    assert matched_row[index["avs_mismatch_flag"]] == 0.0
    assert matched_row[index["billing_shipping_mismatch"]] == 0.0
    assert matched_row[index["cvv_mismatch_flag"]] == 0.0
    assert matched_row[index["is_cross_border"]] == 0.0


def test_ground_truth_attributions_follow_the_recorded_scale():
    """The eleven columns are read against their own ground-truth keys, in contract order."""
    columns = list(feature_names())
    keys = ground_truth_keys()

    probability = {key: 0.1 for key in keys.values()}
    log_odds = {key: -2.0 for key in keys.values()}
    record = {
        "analytical_shapley_probability": probability,
        "analytical_shapley_log_odds": log_odds,
    }

    assert ground_truth_attributions(record, "probability") == [0.1] * len(columns)
    assert ground_truth_attributions(record, "log_odds") == [-2.0] * len(columns)
    # "auto" prefers the exact Owen decomposition, falling back to the quadrature.
    assert ground_truth_attributions(record, "auto") == [-2.0] * len(columns)
    assert ground_truth_attributions(
        {"analytical_shapley_probability": probability}, "auto"
    ) == [0.1] * len(columns)
    # A record with no attributions scores zero rather than raising.
    assert ground_truth_attributions({}, "auto") == [0.0] * len(columns)


def test_benchmark_module_does_not_keep_a_second_copy_of_the_columns():
    """The exported vocabulary lives in one place: `causal_contract`."""
    assert not hasattr(benchmark, "CANONICAL_ATTACK_INTERVENTIONS")
    assert not hasattr(benchmark, "FEATURE_SPECS")
    assert not hasattr(benchmark, "FEATURE_NAMES")


def test_phi_star_is_zero_off_the_declared_support():
    """The intervention vector never credits a column a playbook did not touch."""
    harness = benchmark.XAIBenchmarkHarness(n_transactions=800, fraud_prevalence=0.08, seed=7)
    _, _, _, records = harness.generate_and_prepare_dataset()

    columns = list(feature_names())
    scored = 0
    for record in records:
        if record.get("is_fraud") != 1:
            continue
        phi = harness._build_interventional_ground_truth(record)
        support = intervention_support(str(record.get("scenario_tag", "")))
        assert len(phi) == len(columns)
        if not support:
            assert not np.any(phi)
            continue
        scored += 1
        assert np.isclose(phi.sum(), 1.0)
        assert {columns[j] for j in np.flatnonzero(phi)} == set(support)
    assert scored > 0
