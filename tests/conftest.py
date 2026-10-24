"""Shared fixtures.

The Indian export fixture writes the three files the generator's CLI writes, with
the same columns and value domains, so loader, split, model and serving tests
never need the generator installed. Fraud is made separable on purpose: a
pipeline that mangled the categoricals would score near the baseline here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from prism import tracks

AUTH_HEADER = [
    "transaction_id", "card_id", "pan_masked", "product_id", "cohort_id", "timestamp_utc",
    "tx_time_seconds", "mti", "stan", "rrn", "auth_code", "response_code", "auth_response_code",
    "pos_entry_mode", "pos_condition_code", "eci", "trans_status_3ds", "vaai_score", "amount",
    "amount_minor", "currency", "available_balance", "credit_limit", "mcc", "merchant_id",
]
GATEWAY_HEADER = [
    "transaction_id", "timestamp_utc", "merchant_id", "mid", "tid", "acquirer_bin",
    "gateway_provider", "client_ip", "asn_type", "ip_distance_from_home_km", "device_canvas_hash",
    "channel_type", "avs_match_code", "cvv_match_flag", "geo_risk_score", "is_cross_border",
]
MASTER_HEADER = ["transaction_id", "card_id", "tx_time_seconds", "amount", "is_fraud", "scenario_tag", "risk_score"]

PRODUCTS = [
    "IN_PROD_SALARIED_PRIME_REWARDS", "IN_PROD_SUPER_PREMIUM_HNI",
    "IN_PROD_ENTRY_FD_BACKED", "IN_PROD_PMJDY_RUPAY_DEBIT",
]
COHORTS = [
    "C1_HOURLY_GIG_WORKER", "C2_FIXED_INCOME_SENIOR", "C3_YOUNG_ADULT_STUDENT",
    "C4_SUBURBAN_FAMILY", "C5_URBAN_TECH_PROFESSIONAL", "C6_SMALL_BUSINESS_OWNER", "C7_LUXURY_AFFLUENT",
]
CHANNELS = ["CP_POS_CHIP", "CP_POS_CONTACTLESS", "CNP_WEB", "CNP_MOBILE"]
ENTRY_MODE = {"CP_POS_CHIP": "051", "CP_POS_CONTACTLESS": "071", "CNP_WEB": "012", "CNP_MOBILE": "102"}
MCCS = ["5411", "5812", "5814", "5311", "5732", "5541", "5912", "6051", "6513", "8398"]
GATEWAYS = ["CASHFREE", "RAZORPAY", "PAYU", "BILLDESK"]
ACQUIRERS = ["400814", "459110", "524180", "607152"]
START = 1_704_067_200.0  # 2024-01-01T00:00:00Z, the generator's own epoch


def write_india_export(
    stem: Path, n: int = 3000, cards: int = 60, fraud_share: float = 0.05, seed: int = 0, days: int = 30
) -> Path:
    rng = np.random.default_rng(seed)
    stem.parent.mkdir(parents=True, exist_ok=True)
    times = np.sort(rng.uniform(0.0, days * 86_400.0, size=n)) + START
    card_index = rng.integers(0, cards, size=n)
    card_product = rng.integers(0, len(PRODUCTS), size=cards)
    card_cohort = rng.integers(0, len(COHORTS), size=cards)
    card_limit = rng.uniform(20_000.0, 300_000.0, size=cards)
    fraud = (rng.random(n) < fraud_share).astype(int)

    auth_rows, gateway_rows, master_rows = [], [], []
    for i in range(n):
        card = int(card_index[i])
        is_fraud = int(fraud[i])
        channel = str(rng.choice(CHANNELS, p=[0.1, 0.1, 0.55, 0.25] if is_fraud else [0.35, 0.3, 0.2, 0.15]))
        cnp = channel.startswith("CNP")
        amount = float(rng.gamma(2.0, 900.0) * (4.0 if is_fraud else 1.0)) + 10.0
        vaai = int(rng.integers(35, 90)) if is_fraud else int(rng.integers(5, 45))
        ip_km = float(rng.uniform(15.0, 90.0)) if is_fraud else float(rng.lognormal(2.5, 0.9))
        asn = str(rng.choice(["residential", "mobile", "datacenter"], p=[0.4, 0.3, 0.3] if is_fraud else [0.7, 0.25, 0.05]))
        avs = "Y" if not cnp else str(rng.choice(["Y", "Z", "N", "A"], p=[0.4, 0.35, 0.2, 0.05] if is_fraud else [0.86, 0.06, 0.03, 0.05]))
        cvv = 1 if not cnp else int(rng.random() < (0.85 if is_fraud else 0.98))
        cross = int(cnp and rng.random() < (0.2 if is_fraud else 0.03))
        eci = "" if not cnp else ("07" if rng.random() < (0.3 if is_fraud else 0.05) else "05")
        geo = min(100, 35 * cross + 20 * (avs == "N") + (30 if asn == "datacenter" else 0))
        txn = f"TX_{i:08d}"
        stamp = f"{int(times[i])}"
        auth_rows.append([
            txn, f"CARD_{card:06d}", "607152******1234", PRODUCTS[card_product[card]], COHORTS[card_cohort[card]],
            stamp, f"{times[i]:.2f}", "0100", f"{i % 1_000_000:06d}", f"{i:012d}", "A12345", "00", "00",
            ENTRY_MODE[channel], "59" if cnp else "00", eci, "N" if eci == "07" else "Y", str(vaai),
            f"{amount:.2f}", str(int(round(amount * 100))), "INR", f"{card_limit[card] * 0.6:.2f}",
            f"{card_limit[card]:.2f}", str(rng.choice(MCCS, p=[0.05, 0.05, 0.05, 0.1, 0.25, 0.05, 0.05, 0.2, 0.15, 0.05] if is_fraud else [0.25, 0.2, 0.15, 0.1, 0.08, 0.08, 0.07, 0.02, 0.03, 0.02])), f"M_{i % 40:05d}",
        ])
        gateway_rows.append([
            txn, stamp, f"M_{i % 40:05d}", f"MID_{i % 40:05d}", f"TID_{i % 40:04d}", str(rng.choice(ACQUIRERS)),
            str(rng.choice(GATEWAYS)), "103.1.2.3" if cnp else "127.0.0.1", asn, f"{ip_km:.2f}", "75b24e33fd814bd8",
            channel, avs, str(cvv), str(geo), "True" if cross else "False",
        ])
        # Two fraud scenarios (small and large tickets) and a reference risk score, as the
        # generator's master view carries them: the harness builds folds and label timelines
        # from these columns and never feeds them to a model.
        scenario = ("INTENT_OMEGA_PROBE" if amount < 4_000.0 else "INTENT_OMEGA_HARVEST") if is_fraud else "ORGANIC_NORMAL"
        reference = float(rng.uniform(0.5, 1.0)) if is_fraud else float(rng.uniform(0.0, 0.6))
        master_rows.append([txn, f"CARD_{card:06d}", f"{times[i]:.2f}", f"{amount:.2f}", str(is_fraud), scenario, f"{reference:.4f}"])

    for suffix, header, rows in (
        ("_auth_stream.csv", AUTH_HEADER, auth_rows),
        ("_gateway_telemetry.csv", GATEWAY_HEADER, gateway_rows),
        (".csv", MASTER_HEADER, master_rows),
    ):
        with open(stem.with_name(stem.name + suffix), "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)

    fraud_rows = int(fraud.sum())
    stem.with_name(stem.name + "_calibration.json").write_text(json.dumps({
        "profile": "rbi-psi-2026-07",
        "generated": {
            "rows": n, "fraud_rows": fraud_rows, "requested_fraud_prevalence": fraud_share,
            "registry_fraud_prevalence": 1.2369e-05, "boost_factor": fraud_share / 1.2369e-05,
            "expected_fraud_rows_at_target": n * 1.2369e-05,
        },
        "targets": [], "out_of_scope": [], "overall": "PASS",
    }, indent=2), encoding="utf-8")
    return stem


@pytest.fixture(scope="session")
def india_stem(tmp_path_factory) -> Path:
    return write_india_export(tmp_path_factory.mktemp("india") / "fixture_in")


@pytest.fixture(scope="session", autouse=True)
def pinned_track_for_the_session():
    """Module-scoped fixtures (a trained deployment, a split) run before any
    function-scoped fixture, so the pin has to exist at session scope too."""
    tracks.set_active("ulb")
    yield


@pytest.fixture(autouse=True)
def no_environment_overrides(monkeypatch):
    """A developer who exported PRISM_DATABASE_URL to run the service must not
    have the suite write into that log, nor PRISM_TRACK change what it tests."""
    monkeypatch.delenv("PRISM_DATABASE_URL", raising=False)
    monkeypatch.delenv("PRISM_TRACK", raising=False)
    monkeypatch.delenv("PRISM_LIVE", raising=False)


@pytest.fixture(autouse=True)
def pinned_track():
    """Every test builds ULB-shaped frames unless it passes a schema itself.

    A deployment's config.local.yaml (one that says track: india) must not
    change what the suite tests, so the active track is pinned before each
    test and pinned again after, whatever a test did to it in between.
    """
    tracks.set_active("ulb")
    yield
    tracks.set_active("ulb")


@pytest.fixture(scope="module")
def india_splits_module(india_stem):
    """The Indian fixture with its velocity features, split time-ordered under the Indian
    track; shared by the zoo and explainer tests of one module."""
    from prism.data.india import load_india
    from prism.data.preprocess import prepare_splits
    from prism.features.velocity import add_velocity_features

    tracks.set_active("india")
    try:
        frame = add_velocity_features(load_india(india_stem))
        return prepare_splits(frame, schema=tracks.INDIA)
    finally:
        tracks.set_active("ulb")
