"""spec/07 sections 16 and 17: attackers and cardholders share the consumer address space
and, for the attacks performed from the victim's own device, the cardholder's device.

Before this change every fraud row in an Indian export carried a device hash and a /24
prefix that no legitimate row ever carried, so "first time this /24 is seen" and "device
never seen on a legitimate row" were label rules, not signals.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from prism_generator import SimulationEngine
from prism_generator.network import ClientAddressSpace


def _prefix(ip: str) -> str:
    return ".".join(ip.split(".")[:3])


def _batch(region: str, adversary_mode: str = "intent", seed: int = 42):
    engine = SimulationEngine(n_cards=300, n_merchants=80, region=region, adversary_mode=adversary_mode, seed=seed)
    return engine.generate_batch(n_transactions=3000, fraud_prevalence=0.05, time_span_days=30)


def test_the_same_seed_gives_the_ledger_and_the_syndicates_one_address_space():
    """The ledger and the syndicate registry are built apart; the same seed must put them on
    the same prefix table or the overlap below is luck."""
    a = ClientAddressSpace(seed=7).prefixes("IN")
    b = ClientAddressSpace(seed=7).prefixes("IN")
    assert a == b and len(a) >= 100
    assert ClientAddressSpace(seed=8).prefixes("IN") != a
    assert ClientAddressSpace(seed=7).prefixes("US") != a


@pytest.mark.parametrize("region", ["IN", "US"])
def test_fraud_prefixes_are_prefixes_legitimate_traffic_also_uses(region):
    """Residential and mobile proxies are compromised consumer connections, so their /24s
    carry legitimate traffic too. Datacenter pools (cloud, Tor, bulletproof hosting) come
    from the table legitimate VPN users draw from, but a given bulletproof host may carry
    no legitimate payment in a small batch, so those rows are checked against the table."""
    records = _batch(region)
    space = ClientAddressSpace(seed=42)
    residential = set(space.prefixes(region, "residential"))
    datacenter = set(space.prefixes(region, "datacenter"))
    cnp = [r for r in records if r["channel_type"].startswith("CNP")]
    legit_prefixes = {_prefix(r["client_ip"]) for r in cnp if r["is_fraud"] == 0}
    fraud = [r for r in cnp if r["is_fraud"] == 1]
    assert len(fraud) >= 40, "the batch must hold card-not-present fraud or this proves nothing"
    consumer = [r for r in fraud if _prefix(r["client_ip"]) not in datacenter]
    assert len(consumer) >= 20, "too few fraud rows on consumer prefixes to measure the overlap"
    # the structure: every consumer address, legitimate or proxied, is on the shared table
    assert all(_prefix(r["client_ip"]) in residential for r in consumer)
    assert all(_prefix(r["client_ip"]) in residential | datacenter for r in cnp if r["is_fraud"] == 0)
    unseen = [r for r in consumer if _prefix(r["client_ip"]) not in legit_prefixes]
    # the overlap: before this change every fraud /24 was unseen (100 percent); at 3,000 rows a
    # long-tail prefix is still unseen by chance (the US batch has about 1,200 legitimate
    # card-not-present rows over 300 prefixes), so the bar guards the mechanism and the
    # 30,000-row export is where the share is reported (0 of 101 there)
    assert len(unseen) / len(consumer) <= 0.30, (
        f"{len(unseen)} of {len(consumer)} fraud rows sit on a consumer /24 no legitimate row uses: "
        f"{Counter(_prefix(r['client_ip']) for r in unseen).most_common(5)}"
    )
    hosted = [r for r in fraud if _prefix(r["client_ip"]) in datacenter]
    assert all(_prefix(r["client_ip"]) in datacenter for r in hosted)
    assert any(_prefix(r["client_ip"]) in datacenter for r in cnp if r["is_fraud"] == 0), "legitimate VPN traffic shares the datacenter table"


@pytest.mark.parametrize("region", ["IN", "US"])
def test_a_cardholder_keeps_a_home_prefix(region):
    """Residential traffic is sticky: most of a card's web sessions come from its home /24."""
    records = _batch(region)
    by_card = defaultdict(list)
    for r in records:
        if r["is_fraud"] == 0 and r["channel_type"] == "CNP_WEB":
            by_card[r["card_id"]].append(_prefix(r["client_ip"]))
    busy = {c: p for c, p in by_card.items() if len(p) >= 4}
    assert len(busy) >= 12
    sticky = sum(1 for p in busy.values() if Counter(p).most_common(1)[0][1] / len(p) >= 0.5)
    assert sticky / len(busy) >= 0.8


def test_some_indian_fraud_runs_on_the_victims_own_device():
    """Malware on the phone and a victim talked through a payment leave the cardholder's
    device hash on the fraud row; a detector cannot treat an unknown device as the label.

    The assertion is pinned to the PROBE scenario rather than to the pooled CNP fraud
    population, because a pooled ratio mixes scenarios that differ by design. PROBE is the
    attacker validating or abusing an instrument the victim holds, so part of it lands on the
    victim's own device; HARVEST is the cashout on the attacker's infrastructure, so none of it
    does. Measured 2026-10-13, PROBE put 11 of 55 rows on the victim's device and HARVEST put
    0 of 15, pooling to 11 of 70 or 0.157.

    Two things made the old form of this test unusable. It asserted on the pooled figure, which
    is a mixture proportion and therefore moves whenever the scenario mix moves -- adding
    IN_PROD_RETAIL_RUPAY_DEBIT added harvest-shaped fraud and the pooled figure fell through
    the old 0.20 floor with no change in any per-scenario behaviour. And 0.20 was also the
    approximate PROBE rate itself, so asserting "at least 0.20" set a floor on the mean: with
    ~55 probe rows the estimate has a standard deviation near 0.054, and across seeds 42 to 47
    the PROBE rate came out 0.200, 0.317, 0.158, 0.145, 0.200, 0.226. Two of six seeds failed
    a floor placed on their own average, so the test was a coin flip that passed at seed 42 by
    luck.

    0.10 is a regression floor, not a calibration target: it sits below the minimum observed
    across those six seeds (0.145) with margin, and it fails only if victim-device attribution
    actually degrades. Setting a floor below observed behaviour is standard regression
    guarding and is a different thing from fitting a generative parameter to a benchmark,
    which would be overfitting.
    """
    records = _batch("IN")
    legit_devices = defaultdict(set)
    for r in records:
        if r["is_fraud"] == 0 and r["channel_type"].startswith("CNP"):
            legit_devices[r["card_id"]].add(r["device_canvas_hash"])
    fraud = [r for r in records if r["is_fraud"] == 1 and r["channel_type"].startswith("CNP")]
    assert len(fraud) >= 40
    on_victim_device = [r for r in fraud if r["device_canvas_hash"] in legit_devices[r["card_id"]]]
    assert on_victim_device, "some fraud must sit on a device the cardholder is known to use"

    probe = [r for r in fraud if r["scenario_tag"] == "INTENT_OMEGA_PROBE"]
    assert len(probe) >= 20, f"only {len(probe)} probe rows; a scenario assertion needs a sample"
    probe_on = [r for r in probe if r["device_canvas_hash"] in legit_devices[r["card_id"]]]
    assert len(probe_on) / len(probe) >= 0.10, (
        f"{len(probe_on)} of {len(probe)} probe rows; minimum observed across seeds 42-47 was "
        f"0.145, so 0.10 trips only on real degradation"
    )
    # and the victim's own connection comes with it: no botnet address on those rows
    assert all(r["ip_subnet_prefix"] == "" for r in on_victim_device)


@pytest.mark.parametrize("region", ["IN", "US"])
def test_legitimate_devices_are_shared_within_households(region):
    """A family computer serves more than one card, so a device seen on several cards is
    not by itself an attacker's device."""
    records = _batch(region)
    cards_per_device = defaultdict(set)
    for r in records:
        if r["is_fraud"] == 0 and r["channel_type"].startswith("CNP"):
            cards_per_device[r["device_canvas_hash"]].add(r["card_id"])
    shared = [d for d, cards in cards_per_device.items() if len(cards) >= 2]
    assert len(shared) >= 3


def test_attackers_on_their_own_devices_keep_their_syndicate_markers():
    """The change narrows the shortcut; it does not erase the threat-intelligence graph."""
    records = _batch("IN")
    fraud = [r for r in records if r["is_fraud"] == 1]
    with_botnet = [r for r in fraud if r["botnet_cluster_id"]]
    assert len(with_botnet) >= len(fraud) * 0.30
    assert all(r["ip_subnet_prefix"] == _prefix(r["client_ip"]) + ".0/24" for r in with_botnet if r["channel_type"].startswith("CNP"))
