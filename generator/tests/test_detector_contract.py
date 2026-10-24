"""spec/19_detector_contract.yaml: the one machine-readable description of what a detector
receives, what it is told afterwards, and what the export views carry.

The generator's column lists and its stream take their fields from the contract, so a
column cannot change without the contract changing, and an export conforms to it by test.
"""

from __future__ import annotations


import yaml

from prism_generator import SimulationEngine
from prism_generator.cli import AUTH_STREAM_COLUMNS, GATEWAY_TELEMETRY_COLUMNS
from prism_generator.contract import CONTRACT_FILE, DetectorContract, load_contract
from prism_generator.spec_loader import _find_spec_dir
from prism_generator.stream import AUTHORISATION_OUTCOME_COLUMNS, LABEL_FIELDS, REQUEST_FIELDS, request_payload


def test_the_contract_file_exists_at_version_one():
    path = _find_spec_dir() / CONTRACT_FILE
    assert path.exists()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["contract_version"] == 1
    assert set(raw["feeds"]) == {"authorisation_request", "authorisation_outcome", "label_event"}
    assert set(raw["export_views"]) == {"auth_stream", "gateway_telemetry"}


def test_the_code_takes_its_fields_from_the_contract():
    contract: DetectorContract = load_contract()
    assert tuple(contract.field_names("authorisation_request")) == REQUEST_FIELDS
    assert tuple(contract.field_names("authorisation_outcome")) == AUTHORISATION_OUTCOME_COLUMNS
    assert tuple(contract.field_names("label_event")) == LABEL_FIELDS
    assert list(contract.export_view("auth_stream")) == AUTH_STREAM_COLUMNS
    assert list(contract.export_view("gateway_telemetry")) == GATEWAY_TELEMETRY_COLUMNS


def test_the_request_feed_is_the_two_views_minus_the_outcome():
    contract = load_contract()
    views = dict.fromkeys(list(contract.export_view("auth_stream")) + list(contract.export_view("gateway_telemetry")))
    outcome = set(contract.field_names("authorisation_outcome"))
    assert contract.field_names("authorisation_request") == [c for c in views if c not in outcome]
    assert outcome <= set(contract.export_view("auth_stream"))


def test_every_field_has_a_type_and_categorical_fields_have_a_domain():
    contract = load_contract()
    for feed in ("authorisation_request", "authorisation_outcome", "label_event"):
        for field in contract.fields(feed):
            assert field.type in {"string", "integer", "number", "boolean"}, (feed, field.name, field.type)
            assert field.description, (feed, field.name)
    request = {f.name: f for f in contract.fields("authorisation_request")}
    for name in ("channel_type", "pos_entry_mode", "eci", "trans_status_3ds", "avs_match_code", "asn_type", "currency", "mti"):
        assert request[name].domain, name


def test_a_generated_batch_conforms_to_the_request_feed():
    contract = load_contract()
    engine = SimulationEngine(n_cards=120, n_merchants=40, region="IN", adversary_mode="intent", seed=11)
    records = engine.generate_batch(n_transactions=600, fraud_prevalence=0.08, time_span_days=7)
    problems = []
    for record in records:
        problems.extend(contract.violations("authorisation_request", request_payload(record)))
    assert problems == [], problems[:10]
    engine = SimulationEngine(n_cards=120, n_merchants=40, region="US", adversary_mode="playbook", seed=12)
    records = engine.generate_batch(n_transactions=600, fraud_prevalence=0.08, time_span_days=7)
    problems = []
    for record in records:
        problems.extend(contract.violations("authorisation_request", request_payload(record)))
    assert problems == [], problems[:10]


def test_violations_are_named_by_field_and_reason():
    contract = load_contract()
    bad = {name: None for name in REQUEST_FIELDS}
    bad.update({"amount": "ten", "channel_type": "TELEPATHY", "mcc": 5411})
    found = contract.violations("authorisation_request", bad)
    assert any("amount" in p and "number" in p for p in found)
    assert any("channel_type" in p and "TELEPATHY" in p for p in found)
    assert any("transaction_id" in p for p in found)
