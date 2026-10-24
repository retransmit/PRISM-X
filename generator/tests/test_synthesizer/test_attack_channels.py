"""spec/07 section 18: fraud reaches the mobile and contactless channels.

The intent optimiser's utility has no channel term, so the first candidate channel won
every tie and intent-mode fraud was card-not-present web (or chip, for a track-2 dump)
and never mobile or contactless, which carry 43 percent of Indian legitimate traffic.
"""

from __future__ import annotations

from collections import Counter

import pytest

from prism_generator import SimulationEngine
from prism_generator.intent import (
    AnalyticalBeliefState,
    CredentialDossier,
    CredentialTier,
    InformationDirectedOptimizer,
)


def _fraud_channels(region: str, adversary_mode: str, seed: int = 42) -> Counter:
    engine = SimulationEngine(n_cards=300, n_merchants=80, region=region, adversary_mode=adversary_mode, seed=seed)
    records = engine.generate_batch(n_transactions=3000, fraud_prevalence=0.05, time_span_days=30)
    return Counter(r["channel_type"] for r in records if r["is_fraud"] == 1)


def _dossier(tier: CredentialTier, **flags) -> CredentialDossier:
    return CredentialDossier(tier=tier, pan="4111", cvv2="123", **flags)


def _belief() -> AnalyticalBeliefState:
    return AnalyticalBeliefState(
        card_id="C1", p_valid=0.95, mu_balance=2000.0, sigma_balance=400.0,
        min_balance=0.0, max_balance=5000.0, acquisition_cost_usd=15.0,
    )


def test_the_optimiser_takes_the_preferred_channel_when_the_dossier_allows_it():
    optimizer = InformationDirectedOptimizer(seed=1)
    dossier = _dossier(CredentialTier.TIER_APP_DEVICE_TOKEN, has_bound_token=True)
    action = optimizer.select_optimal_action(_belief(), dossier, current_hour_local=2, preferred_channel="CP_POS_CONTACTLESS")
    assert action.channel == "CP_POS_CONTACTLESS"
    action = optimizer.select_optimal_action(_belief(), dossier, current_hour_local=2, preferred_channel="CNP_MOBILE")
    assert action.channel == "CNP_MOBILE"


def test_the_preferred_channel_never_overrides_a_hardware_constraint():
    optimizer = InformationDirectedOptimizer(seed=1)
    dossier = _dossier(CredentialTier.TIER_CNP_FULLZ)
    action = optimizer.select_optimal_action(_belief(), dossier, current_hour_local=2, preferred_channel="CP_POS_CONTACTLESS")
    assert action.channel in ("CNP_WEB", "CNP_MOBILE")


@pytest.mark.parametrize("region", ["IN", "US"])
def test_intent_fraud_uses_mobile_and_contactless(region):
    channels = _fraud_channels(region, "intent")
    cnp = channels["CNP_WEB"] + channels["CNP_MOBILE"]
    assert cnp >= 60
    assert channels["CNP_MOBILE"] / cnp >= 0.25, dict(channels)
    assert channels["CP_POS_CONTACTLESS"] >= 3, dict(channels)
    assert "CP_CONTACTLESS_NFC" not in channels and "PROVISION_DIGITAL_WALLET" not in channels


@pytest.mark.parametrize("region", ["IN", "US"])
def test_playbook_fraud_uses_mobile(region):
    channels = _fraud_channels(region, "playbook")
    cnp = channels["CNP_WEB"] + channels["CNP_MOBILE"]
    assert cnp >= 60
    assert channels["CNP_MOBILE"] / cnp >= 0.25, dict(channels)


def test_entry_modes_of_mobile_and_contactless_appear_on_fraud_rows():
    """071 (contactless) and 102 (mobile) were legitimate-only values in the export."""
    engine = SimulationEngine(n_cards=300, n_merchants=80, region="IN", adversary_mode="intent", seed=42)
    records = engine.generate_batch(n_transactions=3000, fraud_prevalence=0.05, time_span_days=30)
    fraud_modes = {r["pos_entry_mode"] for r in records if r["is_fraud"] == 1}
    assert {"071", "102"} <= fraud_modes, fraud_modes
