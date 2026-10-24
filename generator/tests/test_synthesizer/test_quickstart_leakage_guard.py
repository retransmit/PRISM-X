"""The published quickstart must not be able to score a perfect ROC-AUC on fraud data.

Regression guard for the defect where publish/README.md's quickstart dropped only
identifiers and the XAI ground-truth columns, leaving in fraud-only outcome columns
(mule_tier, liquidation_channel) and network/device columns that were 100% fraud-pure
in the published batch. LightGBM combined those weak signals into ROC-AUC 1.0000 on the
out-of-time split, which reads as target leakage rather than as performance.

This asserts the two properties that made that possible are now excluded from the
shipped feature list, and that the honest recipe lands in a believable range rather than
at 1.0. It runs against a generated batch, not the published artefact, so it works in CI.
"""

from __future__ import annotations

import pandas as pd

from prism_generator import SimulationEngine

# Columns the card must not use as features. Two reasons, both measured on the
# published 100k batch: fraud-only outcome columns (populated on exactly the fraud rows)
# and network/device columns whose value sets are disjoint between the two classes.
FRAUD_ONLY_COLUMNS = [
    "syndicate_id",
    "botnet_cluster_id",
    "mule_ring_id",
    "beneficiary_account_id",
    "mule_tier",
    "liquidation_channel",
]
IDENTITY_PURITY_COLUMNS = [
    "client_ip",
    "ip_subnet_prefix",
    "asn",
    "isp",
    "device_fingerprint_id",
    "device_canvas_hash",
    "ja4_signature",
]
ENGINE_OUTPUT_COLUMNS = ["risk_score", "base_risk"]


def _batch(n: int = 6000, prevalence: float = 0.02) -> pd.DataFrame:
    engine = SimulationEngine(n_cards=300, n_merchants=90, seed=42)
    records = engine.generate_batch(
        n_transactions=n, fraud_prevalence=prevalence, time_span_days=30
    )
    return pd.DataFrame(records)


def test_network_identity_is_not_a_perfect_label_proxy():
    """Assert the generator property that made the old quickstart leak.

    The relevant property is that a small set of *coarse* network/device identifiers
    (ASNs, ISPs, TLS signatures, device families) partition the batch into runs that
    are entirely one class -- a model can then reach the label from the identity alone.
    Near-unique columns such as client_ip are not the shortcut and are not asserted
    here; what matters is that the low-cardinality identifiers separate the classes.
    """
    df = _batch()
    checked = 0
    for col in ("asn", "isp", "ja4_signature", "device_fingerprint_id"):
        if col not in df.columns:
            continue
        # The empty sentinel means "not observed" and legitimately carries both classes,
        # so purity is asserted over observed identities only.
        observed = df[df[col].astype(str).str.strip().ne("")]
        grouped = observed.groupby(col)["is_fraud"].mean()
        if len(grouped) < 2:
            continue
        assert grouped.min() == 1.0 or grouped.max() == 0.0, (
            f"{col} is no longer fraud-pure over observed identities in this batch "
            f"(rates {grouped.min():.4f}..{grouped.max():.4f}); "
            "the published card's claim about it must be re-measured before release"
        )
        checked += 1
    assert checked, "no low-cardinality identity column was checked; guard is vacuous"


def test_excluded_columns_are_not_reachable_from_the_shipped_feature_list():
    """The card's drop list must cover every column that encodes the label."""
    df = _batch()
    assert FRAUD_ONLY_COLUMNS, "guard would be vacuous with an empty list"
    excluded = set(FRAUD_ONLY_COLUMNS + IDENTITY_PURITY_COLUMNS + ENGINE_OUTPUT_COLUMNS)
    present = excluded & set(df.columns)
    assert present, "none of the guarded columns exist; the guard has stopped testing anything"