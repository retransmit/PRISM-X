"""Tests for geodesic precision, antipodal stability, and physical invariants."""

import math
import pytest
from prism_generator.invariants import (
    EARTH_RADIUS_KM,
    haversine_distance_km,
    verify_transaction_invariants,
)


def test_haversine_identical_points():
    """Haversine distance between identical coordinates must be exactly 0.0."""
    d = haversine_distance_km(40.7580, -73.9855, 40.7580, -73.9855)
    assert d == pytest.approx(0.0, abs=1e-7)


def test_haversine_antipodal_stability():
    """Haversine metric must not raise ValueError: math domain error on antipodal points."""
    # Antipodal points: (0, 0) and (0, 180) -> half earth circumference pi * R
    expected_half_circ = math.pi * EARTH_RADIUS_KM
    d = haversine_distance_km(0.0, 0.0, 0.0, 180.0)
    assert d == pytest.approx(expected_half_circ, rel=1e-4)

    # Extreme polar antipodes: North pole (90, 0) to South pole (-90, 0)
    d_poles = haversine_distance_km(90.0, 0.0, -90.0, 0.0)
    assert d_poles == pytest.approx(expected_half_circ, rel=1e-4)

    # Near-antipodal float rounding test (where a would exceed 1.0)
    d_round = haversine_distance_km(45.0, 0.0, -45.0, 180.0)
    assert d_round == pytest.approx(expected_half_circ, rel=1e-4)


def test_verify_invariants_valid_record():
    """A valid transaction record must pass with zero violations."""
    tx = {
        "transaction_id": "TX_00000001",
        "card_id": "CARD_000001",
        "merchant_id": "M_5411_00001",
        "timestamp_utc": "2024-01-01T12:00:00+00:00",
        "tx_time_seconds": 1704110400.0,
        "amount": 45.50,
        "currency": "USD",
        "mcc": 5411,
        "channel_type": "CP_POS_CHIP",
        "merchant_lat": 40.75,
        "merchant_lon": -73.98,
        "is_fraud": 0,
        "scenario_tag": "ORGANIC_NORMAL",
    }
    violations = verify_transaction_invariants(tx)
    assert len(violations) == 0


def test_verify_invariants_detects_bad_data():
    """Invariant verifier must flag invalid amount, illegal lat/lon, and bad labels."""
    tx_bad = {
        "transaction_id": "TX_BAD",
        "card_id": "CARD_01",
        "merchant_id": "M_01",
        "timestamp_utc": "2024-01-01T12:00:00+00:00",
        "tx_time_seconds": 100.0,
        "amount": -50.0,  # Negative
        "currency": "USD",
        "mcc": 5411,
        "channel_type": "UNKNOWN_CHANNEL",
        "merchant_lat": 95.0,  # Invalid lat
        "merchant_lon": -73.98,
        "is_fraud": 2,  # Non-binary
        "scenario_tag": "NORMAL",
    }
    violations = verify_transaction_invariants(tx_bad)
    assert any("NON_POSITIVE_AMOUNT" in v for v in violations)
    assert any("COORDINATE_LAT_OUT_OF_BOUNDS" in v for v in violations)
    assert any("INVALID_LABEL" in v for v in violations)
    assert any("INVALID_CHANNEL" in v for v in violations)
