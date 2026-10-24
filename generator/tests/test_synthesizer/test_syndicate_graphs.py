"""Tests verifying persistent cybercrime syndicate graph topologies and cross-account linkage."""

from prism_generator import (
    SimulationEngine,
    SyndicateRegistry,
)


def test_syndicate_registry_initialization():
    """Verifies registry creates grounded persistent syndicates for US and IN regions."""
    reg_us = SyndicateRegistry(region="US", seed=42)
    assert len(reg_us.syndicates) >= 3
    us_syn_ids = {s.syndicate_id for s in reg_us.syndicates}
    assert "SYN_US_CARDING_BOT" in us_syn_ids
    assert "SYN_US_SYNTHETIC_BUSTOUT" in us_syn_ids
    assert "SYN_US_ATO_SYNDICATE" in us_syn_ids

    reg_in = SyndicateRegistry(region="IN", seed=42)
    assert len(reg_in.syndicates) >= 3
    in_syn_ids = {s.syndicate_id for s in reg_in.syndicates}
    assert "SYN_IN_MEWAT_VISH" in in_syn_ids
    assert "SYN_IN_INTL_BYPASS" in in_syn_ids
    assert "SYN_IN_RENT_DRAIN" in in_syn_ids


def test_fraud_events_linked_to_syndicates():
    """Verifies fraudulent transactions link to syndicates while normal transactions remain unlinked."""
    engine = SimulationEngine(n_cards=200, n_merchants=60, seed=123)
    records = engine.generate_batch(n_transactions=1000, fraud_prevalence=0.06)

    fraud_records = [r for r in records if r["is_fraud"] == 1]
    normal_records = [r for r in records if r["is_fraud"] == 0]

    assert len(fraud_records) >= 20
    assert len(normal_records) > 800

    # Normal transactions must never leak syndicate metadata
    for r in normal_records:
        assert r["syndicate_id"] == ""
        assert r["botnet_cluster_id"] == ""
        assert r["mule_ring_id"] == ""
        assert r["beneficiary_account_id"] == ""
        assert r["ip_subnet_prefix"] == ""
        assert r["device_fingerprint_id"] == ""

    # Fraud records must have syndicate affiliations
    syn_attributed = [r for r in fraud_records if r["syndicate_id"]]
    assert len(syn_attributed) >= len(fraud_records) * 0.70


def test_mule_ring_cross_card_convergence():
    """Verifies graph topology: multiple distinct victim cards converge to shared mule beneficiary accounts."""
    engine = SimulationEngine(n_cards=300, n_merchants=80, seed=999)
    records = engine.generate_batch(n_transactions=2000, fraud_prevalence=0.08)

    # Group victim cards by beneficiary account
    mule_to_cards = {}
    for r in records:
        b_acc = r.get("beneficiary_account_id", "")
        if b_acc:
            mule_to_cards.setdefault(b_acc, set()).add(r["card_id"])

    assert len(mule_to_cards) > 0

    # There must exist at least one mule account receiving fraudulent funds from multiple cards
    shared_mules = {acc: cards for acc, cards in mule_to_cards.items() if len(cards) >= 2}
    assert len(shared_mules) > 0, "No multi-victim mule convergence detected in bipartite graph!"
