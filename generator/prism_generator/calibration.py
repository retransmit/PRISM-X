"""Calibration report against the public targets in spec/08_india_calibration_targets.yaml.

The report puts every generated batch next to the numbers the regulator publishes: what was
requested, what the registry rate would have given, what the batch actually shows, and for
each target whether it is gated (PASS or FAIL against its tolerance) or only reported and why.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .spec_loader import CalibrationProfileSpec, CalibrationTargetSpec, SpecRegistry, load_all_specs


def _is_credit_product(category: str) -> bool:
    return "CREDIT" in category or "REVOLVING" in category


def _is_debit_product(category: str) -> bool:
    return "DEBIT" in category


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _share_above(amounts: np.ndarray, threshold: float, by_value: bool) -> Optional[float]:
    if amounts.size == 0:
        return None
    above = amounts > threshold
    if by_value:
        total = float(amounts.sum())
        return float(amounts[above].sum() / total) if total > 0.0 else None
    return float(above.mean())


def observed_statistics(records: List[Dict[str, Any]], specs: SpecRegistry) -> Dict[str, Optional[float]]:
    """Every statistic a target can name, measured on the batch. None where the batch cannot say."""
    n = len(records)
    fraud = [r for r in records if int(r.get("is_fraud", 0)) == 1]
    legit = [r for r in records if int(r.get("is_fraud", 0)) == 0]
    fraud_amounts = np.array([float(r["amount"]) for r in fraud], dtype=float)
    total_amount = float(sum(float(r["amount"]) for r in records))

    credit_tickets: List[float] = []
    debit_tickets: List[float] = []
    for r in legit:
        prod_id = str(r.get("product_id", ""))
        product = specs.indian_products.get(prod_id) or specs.products.get(prod_id)
        category = product.category if product else ""
        if _is_credit_product(category) or "CREDIT" in prod_id:
            credit_tickets.append(float(r["amount"]))
        elif _is_debit_product(category) or "DEBIT" in prod_id:
            debit_tickets.append(float(r["amount"]))

    approved_count = sum(1 for r in records if str(r.get("auth_response_code", r.get("response_code", ""))) in ("00", "APPROVED"))
    approval_rate = (approved_count / n) if n else None

    cnp_fraud_amt = sum(float(r["amount"]) for r in fraud if str(r.get("channel_type", "")).startswith("CNP"))
    cnp_fraud_share = (cnp_fraud_amt / float(fraud_amounts.sum())) if fraud_amounts.size > 0 and float(fraud_amounts.sum()) > 0 else None

    return {
        "fraud_prevalence": (len(fraud) / n) if n else None,
        "fraud_value_share_bps": (float(fraud_amounts.sum()) / total_amount * 10_000.0) if total_amount > 0.0 else None,
        "mean_fraud_amount_inr": _mean(list(fraud_amounts)),
        "mean_fraud_amount_usd": _mean(list(fraud_amounts)),
        "fraud_cases_above_10000_share": _share_above(fraud_amounts, 10_000.0, by_value=False),
        "fraud_value_above_10000_share": _share_above(fraud_amounts, 10_000.0, by_value=True),
        "fraud_value_above_50000_share": _share_above(fraud_amounts, 50_000.0, by_value=True),
        "mean_credit_card_ticket_inr": _mean(credit_tickets),
        "mean_debit_card_ticket_inr": _mean(debit_tickets),
        "mean_credit_card_ticket_usd": _mean(credit_tickets),
        "mean_debit_card_ticket_usd": _mean(debit_tickets),
        "authorization_approval_rate": approval_rate,
        "cnp_fraud_value_share": cnp_fraud_share,
    }


def _within_tolerance(target: CalibrationTargetSpec, observed: float) -> Optional[bool]:
    if target.tolerance_relative is not None:
        return abs(observed - target.target) <= target.tolerance_relative * target.target
    if target.tolerance_absolute is not None:
        return abs(observed - target.target) <= target.tolerance_absolute
    return None


def _evaluate(
    target: CalibrationTargetSpec, observed: Optional[float], fraud_rows: int
) -> Dict[str, Any]:
    within = _within_tolerance(target, observed) if observed is not None else None
    if observed is None:
        status = "REPORTED"
        note = "the batch holds nothing to measure this on"
    elif not target.gate:
        status = "REPORTED"
        note = target.reason_not_gated
    else:
        status = "PASS" if within else "FAIL"
        note = ""

    # The verdict is always reported, even when the batch is too small to support it: hiding a
    # failure behind a sample-size rule would let an underpowered batch look clean. The warning
    # says the verdict may be the sample's rather than the generator's.
    reliability_warning = ""
    reliability_basis = target.reliability_basis
    if target.reliable_from_fraud_rows is not None and fraud_rows < target.reliable_from_fraud_rows:
        reliability_warning = (
            f"{target.id} was graded {status} on {fraud_rows} fraud rows, below the "
            f"{target.reliable_from_fraud_rows} this statistic needs before that verdict means "
            f"anything; treat it as indicative, not a result. Derivation in the JSON report "
            f"under reliability_basis."
        )

    return {
        "id": target.id,
        "statistic": target.statistic,
        "target": target.target,
        "observed": observed,
        "tolerance_relative": target.tolerance_relative,
        "tolerance_absolute": target.tolerance_absolute,
        "within_tolerance": within,
        "gate": target.gate,
        "status": status,
        "note": note,
        "source": target.source,
        "reliable_from_fraud_rows": target.reliable_from_fraud_rows,
        "reliability_warning": reliability_warning,
        "reliability_basis": reliability_basis,
    }


def calibration_report(
    records: List[Dict[str, Any]],
    profile_id: str,
    requested_fraud_prevalence: float,
    specs: Optional[SpecRegistry] = None,
) -> Dict[str, Any]:
    """Compares a generated batch with a calibration profile.

    Raises ValueError when the batch's currency does not belong to the profile's region, so a
    US export can never be certified against Indian targets by accident.
    """
    specs = specs or load_all_specs()
    if profile_id not in specs.calibration_profiles:
        raise ValueError(f"Unknown calibration profile {profile_id!r}; known: {sorted(specs.calibration_profiles)}")
    profile: CalibrationProfileSpec = specs.calibration_profiles[profile_id]

    currencies = {str(r.get("currency", "")) for r in records}
    expected_currency = "INR" if profile.region == "IN" else "USD"
    if records and currencies != {expected_currency}:
        raise ValueError(
            f"Profile {profile_id} is for region {profile.region} ({expected_currency}); the batch carries {sorted(currencies)}"
        )

    stats = observed_statistics(records, specs)
    n = len(records)
    fraud_rows = sum(int(r.get("is_fraud", 0)) for r in records)
    targets = [_evaluate(t, stats.get(t.id), fraud_rows) for t in profile.targets]
    gated_failures = [t["id"] for t in targets if t["gate"] and t["status"] == "FAIL"]
    warnings = [t["reliability_warning"] for t in targets if t["reliability_warning"]]

    return {
        "profile": profile.id,
        "region": profile.region,
        "as_of": profile.as_of,
        "scope": profile.scope,
        "generated": {
            "rows": n,
            "fraud_rows": fraud_rows,
            "requested_fraud_prevalence": float(requested_fraud_prevalence),
            "registry_fraud_prevalence": profile.fraud_prevalence,
            "registry_fraud_prevalence_range_2025_26": list(profile.fraud_prevalence_range),
            "boost_factor": float(requested_fraud_prevalence) / profile.fraud_prevalence,
            "expected_fraud_rows_at_target": n * profile.fraud_prevalence,
        },
        "targets": targets,
        "out_of_scope": [dict(item) for item in profile.out_of_scope],
        "overall": "FAIL" if gated_failures else "PASS",
        "gated_failures": gated_failures,
        "warnings": warnings,
    }


def format_report(report: Dict[str, Any]) -> str:
    """A plain-text rendering for the terminal."""
    g = report["generated"]
    lines = [
        f"Calibration profile {report['profile']} ({report['region']}, as of {report['as_of']})",
        f"  rows {g['rows']}, fraud rows {g['fraud_rows']}, requested prevalence {g['requested_fraud_prevalence']:.3g}, "
        f"registry {g['registry_fraud_prevalence']:.3g}, boost x{g['boost_factor']:.0f}, "
        f"fraud rows expected at the registry rate {g['expected_fraud_rows_at_target']:.3f}",
    ]
    for t in report["targets"]:
        observed = "n/a" if t["observed"] is None else f"{t['observed']:.4g}"
        lines.append(f"  {t['status']:8s} {t['id']:34s} target {t['target']:<10.4g} observed {observed:<10s} {t['note']}")
    for w in report.get("warnings", []):
        lines.append(f"  WARNING {w}")
    lines.append(f"  overall {report['overall']}")
    return "\n".join(lines)
