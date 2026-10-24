"""The telemetry of a row agrees with itself (spec/07 section 20).

Three things a detector could use as a rule rather than a signal: an ASN type drawn apart
from the network the address sits on, a device hash on a card-present payment, and master
records whose dictionary fields serialise differently from one machine to the next.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from prism_generator import SimulationEngine, SyndicateRegistry

PROXY_TO_ASN = {"DATACENTER_ROTATING": "datacenter", "MOBILE_4G_5G": "mobile", "RESIDENTIAL_STICKY": "residential"}


def _batch(region: str, seed: int = 42):
    engine = SimulationEngine(n_cards=300, n_merchants=80, region=region, adversary_mode="intent", seed=seed)
    return engine.generate_batch(n_transactions=3000, fraud_prevalence=0.05, time_span_days=30)


@pytest.mark.parametrize("region", ["IN", "US"])
def test_the_asn_type_of_a_proxied_fraud_row_is_the_network_of_its_proxy(region):
    records = _batch(region)
    registry = SyndicateRegistry(region=region, seed=42)
    proxy_kind = {b.cluster_id: b.proxy_type for s in registry.syndicates for b in s.botnets}
    proxied = [r for r in records if r["is_fraud"] == 1 and r["channel_type"].startswith("CNP") and r["botnet_cluster_id"]]
    assert len(proxied) >= 30, "the batch must hold proxied card-not-present fraud or this proves nothing"
    wrong = Counter(
        (proxy_kind[r["botnet_cluster_id"]], r["asn_type"])
        for r in proxied
        if r["asn_type"] != PROXY_TO_ASN[proxy_kind[r["botnet_cluster_id"]]]
    )
    assert not wrong, f"ASN types that contradict the proxy's network: {dict(wrong)}"


@pytest.mark.parametrize("region", ["IN", "US"])
def test_a_card_present_payment_carries_no_device_hash(region):
    records = _batch(region)
    card_present = [r for r in records if r["channel_type"].startswith("CP")]
    card_not_present = [r for r in records if r["channel_type"].startswith("CNP")]
    assert any(r["is_fraud"] == 1 for r in card_present)
    assert all(r["device_canvas_hash"] == "" for r in card_present)
    assert all(len(r["device_canvas_hash"]) == 16 for r in card_not_present)


def test_dictionary_fields_of_a_record_serialise_as_plain_json():
    engine = SimulationEngine(n_cards=60, n_merchants=20, region="IN", adversary_mode="intent", seed=5)
    records = engine.generate_batch(n_transactions=400, fraud_prevalence=0.08, time_span_days=10)
    for record in records:
        for field in ("counterfactual_twin", "normative_baseline", "counterfactual_input_deltas"):
            value = record[field]
            assert "np." not in repr(value), (field, repr(value)[:120])
            json.dumps(value)
