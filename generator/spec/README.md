# Prism Generator Living Specification (`spec/`)

This directory is the **canonical Single Source of Truth (SSOT)** for the Prism Generator simulation platform.

## Purpose
To eliminate the gap between high-level research and executable reality, all domain knowledge (banking rules, human psychology, merchant friction, and fraudster playbooks) must be codified into typed specifications here **before** being implemented in code.

## Structure
- [`active_slice.yaml`](active_slice.yaml): The operational focus tracker. Read first in any session.
- `01_financial_instruments.yaml`: Card product tiers (Subprime, Prime, Ultra-High-Net-Worth, Corporate P-Cards, Debit, Virtual), limits, auth flows, and rail vulnerabilities.
- `02_human_personas.yaml`: Human lifestyle archetypes, circadian rhythms, notification fatigue, discovery latencies, and life quirks.
- `03_payment_rail_gaps.yaml`: Payment plumbing details (ISO 8583 codes, STIP, AVS, 3DS exemptions, dual-message settlement).
- `04_adversarial_playbooks.yaml`: Cybercrime playbooks (card testing, ATO baking, bust-out, BIN attacks, amount decay).
- `05_india_payment_rails.yaml`: RBI AFA and card-control defaults, RuPay on UPI, limited-liability tiers, CFCFRMS 1930 race, Indian playbooks.
- `06_adversarial_intent_mesh.yaml`: Credential completeness tiers (Fullz, Phished OTP, Session Cookies, Track-2 Dumps), supply chains, and multi-hop switch topology.
- `07_export_leakage_gate.yaml`: Export-level label leakage gate, anti-separability gates, shared /24 prefix pools, victim-device rules, and prevalence pacing.
- `08_india_calibration_targets.yaml`: Dated calibration profiles for the Indian card stream (RBI PSI fraud prevalence and ticket sizes).
- `09_us_calibration_targets.yaml`: Federal Reserve Payments Study, Visa approval rate benchmarks, and DCPC ticket distributions.
- `16_operational_supervision.yaml`: Operational triage queues, Weibull analyst latencies, and LogNormal chargeback dispute arrival lags.
- `17_prequential_evaluation.yaml`: Prequential time-ordered evaluation, rolling retraining, and streaming drift auditing under delayed supervision.
- `18_benchmark_reporting.yaml`: Four-pillar certification thresholds and publication-grade reporting schema.
- `19_detector_contract.yaml`: The detector contract (v1): pre-decision authorization feed and gateway telemetry view definitions guaranteeing zero label leakage.

## Rules of Engagement
1. **Never write code without a spec.**
2. **Never put concepts in a spec that cannot run in Python/NumPy.**
3. **Every spec item must have test verification in `tests/`.**
