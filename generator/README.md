# Prism Generator: Grounded Multi-Agent Payment Fraud Simulation & Causal XAI Benchmark

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](pyproject.toml)
[![Invariants: 37/37 Verified](https://img.shields.io/badge/invariants-37%2F37%20verified-brightgreen.svg)]()
[![Rails: ISO 8583 | RBI AFA | Visa VCR](https://img.shields.io/badge/rails-ISO%208583%20%7C%20RBI%20AFA%20%7C%20Visa%20VCR-orange.svg)]()
[![Contract: Spec 19 v1](https://img.shields.io/badge/contract-Spec%2019%20v1-purple.svg)](spec/19_detector_contract.yaml)
[![Anti-Astronaut Certified](https://img.shields.io/badge/grounding-Anti--Astronaut%20Certified-darkgreen.svg)](AGENTS.md)

**Prism Generator** is an open-source, publication-grade multi-agent payment fraud simulation framework and causal explainable AI (XAI) benchmarking platform. It models high-throughput financial switches under continuous physical time, authentic dual-region banking rails (**United States** and **India**), and adaptive cybercrime syndicates.

Unlike legacy synthetic datasets that rely on static, ungrounded tabular distributions (such as PaySim or 2013 PCA benchmarks), Prism Generator couples event synthesis with a closed-form **Structural Causal Model (SCM)** to generate mathematically exact Pearlian counterfactual twins ($\Delta \mathbf{x} = \mathbf{x}_{\text{fraud}} - \mathbf{x}_{\text{baseline}}$) and analytical Shapley attributions ($\phi^*$). This provides an objective, auditable ground-truth baseline to benchmark post-hoc explainers (TreeSHAP and a counterfactual-twin occlusion baseline) under realistic streaming delayed feedback.

---

## The Two Crises in Financial Fraud AI

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  1. THE GROUND-TRUTH EXPLANATION VOID                            │
│  Production banking warehouses record binary dispute outcomes (Y ∈ {0, 1}), never the true causal│
│  reasons why an authorization occurred. When banks deploy post-hoc XAI tools (TreeSHAP, LIME),   │
│  compliance teams have zero ground truth to verify if explanations reflect authentic fraud       │
│  mechanisms or dangerous heuristic artifacts, violating ECOA Reg B and SR 11-7 requirements.     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 ▲
                                                 │ Prism Generator bridges both gaps
                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               2. THE REALISM & SUPERVISION LATENCY GAP                           │
│  Real transaction logs are confidential under PCI-DSS. Meanwhile, real fraud supervision is      │
│  severely delayed: chargeback disputes lag by 3 to 120 days, investigator queues have finite     │
│  daily capacities (Top-K alerts), and small-ticket fraud goes unreported (dark fraud).           │
│  Models trained on stale, static tabular snapshots suffer catastrophic concept drift.            │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Capabilities & Architectural Pillars

### 1. The Formal Detector Contract (`spec/19_detector_contract.yaml` & `contract.py`)
Prism Generator establishes an immutable, machine-readable contract (**version 1**) governing all interactions between the simulation and external detection models. The contract guarantees zero target label leakage by decoupling streaming transactions into three distinct feeds:
* **Authorisation Request Feed (35 fields):** The exact pre-decision data an acquirer or payment gateway passes to an issuer risk switch in sub-50ms (combining `auth_stream` and `gateway_telemetry` on `transaction_id`). Strictly excludes issuer outcomes, scenario tags, and internal generator scores (`NEVER_IN_A_REQUEST`).
* **Authorisation Outcome Feed (3 fields):** The issuer host's decision (`response_code`, `auth_response_code`, `auth_code`), released only *after* the detector has scored the request.
* **Label Event Feed (7 fields):** The supervision verdict (`INVESTIGATOR_ALERT` within 0.5-72 hours, `CHARGEBACK_DISPUTE` within 3-120 days, or `UNLABELLED` for dark fraud), released strictly in chronological discovery order.

All CLI exports, storage formats, and streaming daemons take their columns directly from `load_contract()`.

### 2. Closed-Loop Streaming Daemon & Active Learning
`python -m prism_generator.stream` posts authorization requests to an external REST endpoint and releases labels on a dedicated feed as the bank would naturally discover them.
* **Detector-Driven Supervision Queue:** When an endpoint responds with a prediction score (`risk_score` or `--score-field`), that score directly determines which transactions enter the bank's daily investigator review budget (top-$K$ alerts).
* **Operational Feedback Loop:** If the detector under test achieves high precision, true frauds are verified and added to the bank's training pool within hours. If the detector has blind spots, fraudulent transactions escape into the 3-120 day chargeback delay or dark fraud pool. The training stream dynamically mirrors the operational performance of the detector under test.

### 3. Dual-Region Payment Rail Kinematics
* **United States Payment Infrastructure (USD Cents):**
  * Models the two-stage lifecycle: Authorization (`MTI 0100` $\to$ `0110`) followed by Financial Presentment/Clearing (`MTI 0200` $\to$ `0210`) with 24-72h settlement delays.
  * Real-world pre-authorization mechanics: Automated Fuel Dispenser (AFD, MCC 5542) \$175 hold vs. final nozzle cutoff; Dining (MCC 5812) post-auth 10-20% tip tolerances; Hotel/Lodging (MCC 7011) incidental folio holds.
  * Stand-In Processing (STIP) floor limits when issuer cores timeout ($> 2.0\,\text{s}$).
  * Address Verification Service (AVS) numeric street + ZIP matrix adjudication (`Y`, `A`, `Z`, `N`, `U`).
* **Indian Payment Rails (INR Paisa):**
  * Mandatory Additional Factor of Authentication (AFA/OTP challenges) on domestic CNP transactions enforcing `ISO 63` declines.
  * Contactless NFC tap-and-pay rules (₹5,000 PIN ceiling, ₹15,000 cumulative velocity cap enforcing `ISO 65`).
  * RuPay Credit Cards on UPI QR code rails (POS Entry Mode `031`).
  * RBI Master Directions (`RBI/2017-18/15`) limited liability customer tiers (Zero Liability, ₹5k/₹10k caps) and CFCFRMS Helpline 1930 golden hour cyber-liens.
  * Product taxonomies: Kisan Credit Card (KCC), PMJDY RuPay Debit, FD-Backed Entry Cards, Salaried Prime Rewards, and Super-Premium HNI cards.

### 4. Shared Client Infrastructure & Anti-Separability (`network.py`)
To prevent machine learning models from exploiting synthetic artifact shortcuts:
* **Unified Telecom IP Prefix Pools:** Cardholders and cybercrime botnets share a deterministic consumer IP address space (`ClientAddressSpace`) governed by Zipf popularity distributions over authentic regional telecom allocations (Reliance Jio and Airtel `49.x`, `103.x`, `106.x` in India; Comcast and AT&T `24.x`, `67.x`, `72.x` in the US). Measured on a 4,000-transaction US batch at 5% fraud (seeds 42 / 7 / 123), 84% / 95% / 100% of fraud records arrived from a `/24` that also carried legitimate traffic, so an address prefix alone does not separate the classes. The prefix field itself (`ip_subnet_prefix`) is populated from syndicate telemetry only, and is exported to the labelled threat-intel enclave rather than to the detector's inference feed (`spec/19`).
* **Victim-Device Execution:** 70% to 95% of vishing (`IN_ADV_REVERSE_PROXY_VISHING`) and Android malware (`IN_ADV_APK_SMS_STEALER`) attacks execute directly on the cardholder's own mobile device and residential IP connection, eliminating naive device-hash separability.
* **Card-Present Terminal Telemetry:** Physical POS terminals and ATMs carry no user device hash (`device_canvas_hash = ""`), while legitimate web/mobile users realistically share household computers and secondary devices.

### 5. Stepped Underwriting Credit Lines & Attack Diversity
* **Discrete Credit Limit Steps:** Credit lines round to authentic issuer underwriting steps (`spec/01` and `spec/05`), replacing continuous random floats with discrete financial tiers (e.g., ₹25k, ₹50k, ₹1L; \$100, \$250, \$500, \$5k). Measured across 1,000 cards: 77 distinct limits in the IN ecosystem and 88 in the US (seed 42), instead of 1,000 distinct random floats.
* **Attack Channel Diversity:** Attacks realistically span Web (`012`), In-App Mobile (`102`), Chip (`051`), and Contactless NFC (`071`).
* **Prevalence Pacing:** A trailing 24-hour window re-estimates competing Poisson attack arrival intensity, ensuring small batches (1,000-5,000 rows) accurately hit the requested fraud prevalence.

### 6. Closed-Form Pearlian Causal Ground Truth ($\phi^*$)
* **Exact Counterfactual Baseline Twins:** For every simulated fraudulent transaction, the engine computes:
  $$\Delta \mathbf{x} = \mathbf{x}_{\text{fraud}} - \mathbf{x}_{\text{baseline}}$$
* **Analytical Shapley Derivation:** Closed-form Owen multilinear extensions in logit space and 128-point path integration in probability space yield exact Shapley values ($\phi^*$).
* **Quantitative Explainer Benchmarking:** Evaluates post-hoc explainers (TreeSHAP on the gradient-boosted detector, plus a counterfactual-twin occlusion baseline) against exact ground truth using formal metrics: Precision@k, Recall@k, Kendall's $\tau_b$, Spearman's $\rho$, and Relative Attribution Error (RAE).

---

## Architecture Overview

```
                                      PRISM GENERATOR SIMULATION & BENCHMARK ARCHITECTURE

  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   DISCRETE-EVENT MONOTONIC SIMULATION                                  │
  │  • 64-bit microsecond priority queue                        • Non-stationary Hawkes Point Processes    │
  │  • Haversine velocity ceilings (< 900 km/h)                 • Circadian spend & arrival thinning       │
  └───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                      │
                         ┌────────────────────────────┴────────────────────────────┐
                         ▼                                                         ▼
  ┌──────────────────────────────────────────────┐          ┌──────────────────────────────────────────────┐
  │              CARDHOLDER AGENTS               │          │             ADAPTIVE RED TEAM                │
  │  • 7 Demographic cohorts (Fed DCPC / RBI)    │          │  • 13 Grounded cybercrime playbooks          │
  │  • Shared Zipf /24 residential IP prefixes   │          │  • Closed-loop adaptation to ISO declines    │
  │  • Household and secondary device bindings   │          │  • Victim-device malware / vishing execution │
  └──────────────────────┬───────────────────────┘          └──────────────────────┬───────────────────────┘
                         │                                                         │
                         └────────────────────────────┬────────────────────────────┘
                                                      ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                      DECOUPLED VERIFIER SWITCH & SCM                                   │
  │  • Dual-message ISO 8583 authorization / clearing           • Structural Causal Model (SCM)            │
  │  • RBI AFA / OTP, CoFT, and contactless caps                • Closed-form Owen Shapley ground truth φ* │
  └───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   SPEC 19: FORMAL DETECTOR CONTRACT                                    │
  │                               (Machine-Readable Single Source of Truth v1)                             │
  └───────────────────┬───────────────────────────────┬───────────────────────────────┬────────────────────┘
                      │                               │                               │
                      ▼                               ▼                               ▼
       [Authorisation Request Feed]      [Authorisation Outcome Feed]      [Delayed Label Event Feed]
       • 35 strictly pre-decision fields • Response code (00, 51, 63)      • Investigator alert (0.5-72h)
       • Joined auth + gateway telemetry • Approval code (auth_code)       • Chargeback dispute (3-120d)
       • ZERO target label leakage       • Revealed ONLY after scoring     • Dark fraud never released
                      │                                                               ▲
                      ▼                                                               │
          ┌───────────────────────┐                                                   │
          │  Detector Under Test  │                                                   │
          │  (External Model/REST)│                                                   │
          └───────────┬───────────┘                                                   │
                      │ answers with risk_score                                       │
                      ▼                                                               │
          ┌───────────────────────────────────────────────────────────────────────────┴───┐
          │                      CLOSED-LOOP OPERATIONAL SUPERVISION                      │
          │  • Daily top-K review budget prioritizes rows scored by the detector under    │
          │    test; learned labels dynamically follow the detector's true triage efficacy│
          └───────────────────────────────────────────────────────────────────────────────┘
```

---

## Four Institutional Banking Feeds

Rather than exporting a single flat table with unrealistic columns, Prism Generator partitions synthetic outputs into the four authentic feeds that production banking data warehouses maintain:

| Feed File | Schema Standard | Description |
| :--- | :--- | :--- |
| **`auth_stream.csv`** | ISO 8583 / ISO 20022 | Real-time authorization switch feed containing MTI, STAN, RRN, Auth Code, Response Code, POS Entry Mode, ECI, 3DS status, available balances, and minor currency units. |
| **`gateway_telemetry.csv`** | Gateway Risk Telemetry | Network & device telemetry containing Client IP, ASN type, geo risk scores, device canvas hashes, AVS codes, CVV match flags, and cross-border flags. |
| **`clearing_settlement.csv`** | Dual-Message Settlement | Financial presentment feed (`MTI 0200`) with 24-72h delay windows, actual captured amounts (AFD pump vs hold, dining tips), and interchange fees. |
| **`dispute_recovery.csv`** | Scheme & Statutory Disputes | Chargeback logs with reason codes (Visa 10.4, Mastercom), Visa CE 3.0 pre-dispute deflection, arbitration fees, and RBI Limited Liability Customer Tiers (`RBI/2017-18/15`). |

---

## Quick Start

### Installation

```bash
cd generator                   # inside a checkout of this repository
pip install -e ".[dev,benchmark,gui]"
```

Or using **[uv](https://github.com/astral-sh/uv)** for fast installation:
```bash
uv pip install -e ".[dev,benchmark,gui]"
```

---

### Interactive Web GUI (Prism Generator Studio)

Prism Generator includes an interactive web interface built with FastAPI and React (TypeScript + Tailwind CSS):

```bash
# Launch the interactive web GUI (serves on http://127.0.0.1:8000)
prism_generator gui --open
```

The interface provides four functional workspaces:
- **Simulation Generator**: Configure payment rails (US dual-message vs. India RBI AFA), transaction volumes, fraud prevalence, and adversary architectures.
- **Transaction Ledger & ISO 8583 Inspector**: Searchable and filterable transaction table with a slide-over drawer showing raw ISO 8583 MTI codes, 3DS telemetry, network risk scores, and device fingerprints.
- **Causal Explainability (XAI)**: Interactive side-by-side bar chart comparing post-hoc TreeSHAP attributions against closed-form SCM ground truth $\phi^*$ and Pearlian counterfactual baseline deltas ($\Delta \mathbf{x}$).
- **Classifier & XAI Benchmarking**: One-click benchmark evaluating LightGBM or Random Forest on ROC-AUC, PR-AUC, Kendall's $\tau_b$, Spearman's $\rho$, and Relative Attribution Error (RAE).

---

### Command-Line Interface (CLI)

#### 1. Synthesize US Dual-Message Transactions
```bash
# Generates 5,000 US transactions partitioned into 4 institutional feeds
prism_generator generate \
    -n 5000 \
    --cards 1000 \
    --merchants 150 \
    --region US \
    --fraud-rate 0.03 \
    --days 30 \
    --seed 42 \
    -o data/us_production.csv \
    --include-disputes \
    --export-institutional-views
```

#### 2. Synthesize Indian Rails (RBI AFA / RuPay / CoFT Calibrated)
```bash
# Calibrated against the RBI Central Payments Fraud Information Registry (July 2026 targets)
prism_generator generate \
    -n 5000 \
    --region IN \
    --calibration rbi-psi-2026-07 \
    --seed 42 \
    -o data/calibrated_india.csv \
    --include-disputes \
    --export-institutional-views
```

#### 3. Stream Authorization Requests to an External Detector
```bash
# Posts authorization requests to an ML endpoint; releases labels on a separate feed
python -m prism_generator.stream \
    --region IN \
    --calibration rbi-psi-2026-07 \
    --duration 600 \
    --tps 10 \
    --endpoint http://localhost:8000/api/v1/predict \
    --label-endpoint http://localhost:8000/api/v1/labels

# Inspect decoupled request and label feeds directly on stdout as JSON lines:
python -m prism_generator.stream --region US --duration 10 --tps 5 --stdout
```

---

### Benchmarking Post-Hoc Explainers Against Causal Ground Truth

Prism Generator includes an automated evaluation harness conforming to **Quantus (JMLR 2023)** and **OpenXAI (NeurIPS 2022)** standards:

```bash
prism_generator benchmark -n 2000 --model lightgbm --seed 42
```

Sample Benchmark Output (`prism_generator benchmark -n 2000 --model lightgbm --seed 42`, reproduced locally; run the same command to regenerate):
```
=================================================================
  PRISM GENERATOR EMPIRICAL XAI BENCHMARK RESULTS
=================================================================
  Model Architecture:           LIGHTGBM
  Explainer Method:             TreeSHAP (Interventional)
  Evaluated Fraud Samples:      41
  Classifier ROC-AUC:           0.9757
  Classifier PR-AUC:            0.8255
-----------------------------------------------------------------
  Ranking Concordance (Kendall Tau):      0.1996
  Rank Correlation (Spearman Rho):        0.2447
  Top-3 Support Recovery (Precision@3):   0.3008
  Intervention Precision (P@3):           0.3761
  Intervention Recall (R@3):              0.3162
  Relative Attribution Error (RAE):       10.1651
=================================================================
```

The continuous-error rows here (RAE, cosine, correlations) compare a margin-scale
TreeSHAP learner against a probability-scale closed-form target, so their absolute
values are operator-limited; the floor-gated row in this harness is P@3 (the continuous
rows carry collapse tripwires only), and the same-object graded attribution rows are the
certification's Pillar 4 below (`docs/PROJECT_CONTEXT.md` 7.1 records the floor
re-derivations).

---

### Four-Pillar Certification Benchmark (the graded run)

The explainer harness above measures XAI only. The **certification** benchmark grades all
four pillars of `spec/18_benchmark_reporting.yaml` and issues the verdict:

```bash
prism_generator benchmark -n 2000 --unified --seed 42 --output-dir reports/benchmark
```

`-n 2000` and `--seed 42` are not suggestions: `spec/18` pins them under
`evaluation_protocol`, together with the region, stream length and streaming windows, and a
list of seeds to repeat the run over. A run that deviates is reported as **OFF-PROTOCOL**
with each deviation named, so its violation count cannot be mistaken for the pinned one.

Because several gates rest on small denominators (5-7 streaming test days at the pinned
shape; ~160 fraud records behind the evasion share), **one seed's violation count is not a
result.** On-protocol runs therefore re-execute every pinned seed and print a per-gate
table instead. Verbatim from the run above (the `#` is where the observed value and the
limit are stripped, so the same gate groups across seeds; 3 gates failed at least once,
all shown):

```
| gate                                                                       | failed in | reading          |
|---                                                                         |---        |---               |
| Pillar 3 (Operational Streaming): prequential PR-AUC # falls below the minimum | 3/5 | FLIPS WITH SEED |
| Pillar 1 (Data Fidelity): MCC Jensen-Shannon divergence # exceeds the maximum | 2/5 | FLIPS WITH SEED |
| Pillar 2 (Privacy & Robustness): evasion rate (approved share of fraud records) # exceeds the maximum | 1/5 | FLIPS WITH SEED |
```

Measured 2026-10-16, the **pinned run itself grades `TIER-1_GOLD` with 0 violations** , 
all 19 gates pass; across the five pinned seeds the grades are `TIER-1_GOLD` x1,
`TIER-2_SILVER` x2 and `NON_CERTIFIED_FAIL` x2 (violation counts 0-2). No gate fails in
every seed, and the three that still flip are the small-denominator movers: prequential
PR-AUC over 5-7 test days, an MCC mix drawn from 25 merchants, and the evasion share
over ~160 fraud records. The full measurement and its reproducing command are in
[`reports/certification_grade_stability.md`](reports/certification_grade_stability.md).

---

### Python Programmatic API

```python
from prism_generator import SimulationEngine, XAIBenchmarkHarness

# 1. Initialize simulation engine for Indian payment ecosystem
engine = SimulationEngine(
    n_cards=500,
    n_merchants=100,
    region="IN",
    seed=42,
)

# 2. Generate a batch of continuous-time transactions
records = engine.generate_batch(
    n_transactions=1000,
    fraud_prevalence=0.04,
    time_span_days=14,
)

sample = records[0]
print(f"TX: {sample['transaction_id']} | MTI: {sample['mti']} | ISO Field 39: {sample['response_code']}")
print(f"Amount: {sample['amount']} {sample['currency']} | Credit Limit: {sample['credit_limit']}")
print(f"Causal Ground Truth Driver: {sample['dominant_causal_driver']}")

# 3. Benchmark TreeSHAP against closed-form SCM ground truth
harness = XAIBenchmarkHarness(n_transactions=1000, fraud_prevalence=0.05, seed=42)
benchmark_summary = harness.run_benchmark(model_type="lightgbm")
print(f"Kendall's Tau Concordance: {benchmark_summary.mean_kendall_tau:.4f}")
print(f"Intervention Precision@3:  {benchmark_summary.mean_intervention_precision_at_3:.4f}")
```

---

## Living Specification Registry (`spec/`)

The simulation is governed by 17 numbered specification files serving as the single source of truth (the active slice is tracked in `spec/active_slice.yaml`):

| Specification File | Scope & Technical Mandate |
| :--- | :--- |
| **`spec/01_financial_instruments.yaml`** | 11 Global card products, issuer credit limit stepping grids, and interchange schedules. |
| **`spec/02_human_personas.yaml`** | 7 Fed DCPC demographic cohorts, Dirichlet spend allocations, and circadian simplexes. |
| **`spec/03_payment_rail_gaps.yaml`** | AFD holds, dining tip tolerances, STIP timeout rules, AVS matrix, and Visa CE 3.0. |
| **`spec/04_adversarial_playbooks.yaml`** | 13 grounded cybercrime playbooks (8 US: micro-auth probing, ATO silent baking, sleeper bust-out, Apple Pay Yellow Path, nocturnal burst, BIN enumeration, triangulation, collusive bust-out; 5 India: reverse-proxy vishing, APK SMS stealer, non-3DS bypass, rent-portal cashout, SIM-swap/eSIM hijack), liquidation channels, and the mule pipeline. |
| **`spec/05_india_payment_rails.yaml`** | RBI AFA/OTP, RuPay on UPI, contactless limits, PMJDY overdraft, and 1930 cyber-liens. |
| **`spec/06_adversarial_intent_mesh.yaml`** | Credential completeness tiers (Fullz, Phished OTP, Session Cookies, Track-2 Dumps), information supply chains, syndicate roles, and multi-hop switch topology. |
| **`spec/07_export_leakage_gate.yaml`** | Anti-separability gates, shared /24 prefix pools, victim-device rules, and prevalence pacing. |
| **`spec/08_india_calibration_targets.yaml`**| RBI Central Payments Fraud Information Registry (CPFIR) empirical targets. |
| **`spec/09_us_calibration_targets.yaml`** | Federal Reserve Payments Study, Visa Q3 FY25 approval rate, and DCPC 2023 ticket distributions. |
| **`spec/13_annual_seasonality_and_drift.yaml`** | Annual macro regime calendar and seasonal drift schedules. |
| **`spec/14_interbank_switch_and_clearing.yaml`** | Multi-bank routing mesh, switch topology, and clearing/settlement rules. |
| **`spec/16_operational_supervision.yaml`**| Operational triage queues, Weibull analyst latencies, and LogNormal chargeback lags. |
| **`spec/17_prequential_evaluation.yaml`** | Prequential time-ordered evaluation, rolling retraining, and streaming drift auditing under delayed supervision. |
| **`spec/18_benchmark_reporting.yaml`** | Camera-ready figure typography and the four-pillar certification thresholds read by the benchmark grader. |
| **`spec/19_detector_contract.yaml`** | **The Detector Contract (v1):** Strict 3-feed schema definitions guaranteeing zero leakage. |
| **`spec/20_backbone_replay.yaml`** | External corpus replay schema: injection over a real traffic backbone with exact pacing and zero label leakage. |
| **`spec/21_prism_causal_evaluation.yaml`** | PRISM two-track causal-XAI evaluation and the normalising-flow quarantine (`ground_truth_authority: physical_intervention_delta`). |

---

## Empirical Verification & Test Results

Prism Generator enforces strict, deterministic verification across the entire stack:

### 1. PyTest Test Suite (**410 / 410 Passed, 100% Green**)
```bash
pytest tests/ -v
```
Certifies:
* **Detector Contract Conformance (`spec/19`):** Authorisation request feeds contain strictly pre-decision fields; zero scenario tags, labels, or internal reference scores leak.
* **Closed-Loop Feedback:** Stream daemon's investigation queue follows the detector under test; labels released in discovery order without temporal lookahead.
* **Anti-Separability:** Attackers and cardholders share consumer IP space; victim-device malware retains cardholder device bindings; credit lines fall on issuer steps.
* **Causal Attribution Efficiency:** Owen multilinear extensions and 128-point path integration satisfy the Shapley efficiency axiom.
* **Kinematics & Invariants:** Antipodal stability, Haversine travel velocities strictly $< 900\,\text{km/h}$, strict temporal monotonicity, and balance conservation across declines.

### 2. Grounded 37-Scenario Operational Invariant Engine (**100% Passed**)
```bash
python scripts/verify_grounded_invariants.py
```
Validates 37 formal operational scenarios spanning:
* **Part A (Adversarial Playbooks):** Micro-auth probing, 14-day silent ATO baking, sleeper bust-outs, Apple Pay yellow-path tokenization, nocturnal bursts, reverse-proxy vishing, and APK SMS stealers.
* **Part B (Payment Rail Plumbing):** AFD \$175 pre-auth nozzle cutoffs, hotel incidental folios, dining 20% tip adjustments, STIP 2.0s SLA outages, AVS matrix adjudication, and India ₹5,000 contactless limits.
* **Part C (Consumer Dynamics & Quirks):** Dhanteras gold splitting (Rule 114B ₹2L cap), fuel surcharge waivers, no-cost EMI discounts, forgotten subscription churn, and multi-modal discovery latencies.
* **Part D (Merchant Risk & Disputes):** PEA additive guessing, triangulation fraud multipliers, Visa CE 3.0 pre-dispute deflection, \$500 network arbitration veto, RBI Circular `RBI/2017-18/15` limited liability tiers, and CFCFRMS Helpline 1930 golden hour races.

### 3. Forensic Dataset Realness Audit
```bash
python scripts/audit_fraud_realness.py
```
* **Benford's Law First-Digit Conformity:** Legitimate transaction amounts conform to Benford's Law (Mean Absolute Deviation $< 0.012$), while fraud spend exhibits anomalous digit manipulation.
* **Circadian Dynamics:** Diurnal arrival schedule suppresses nocturnal transactions to $< 4.5\%$ while peaking during retail business hours.
* **Realistic ML Separability:** Logistic Regression PR-AUC falls in the realistic operational range ($0.30 - 0.50$), proving that hard negatives create genuine false alarms and stealthy fraud slips through.

---

## Anti-Astronaut Grounding Mandate

This repository adheres strictly to the **Anti-Astronaut Grounding Mandate** codified in [`AGENTS.md`](AGENTS.md):
1. **Zero Theoretical Buzzwords:** No speculative, non-implementable concepts (no ungrounded quantum algorithms or non-executable claims).
2. **Physical & Financial Units:** Every parameter has explicit units (USD cents, INR paisa, seconds, km/h, probabilities in $[0.0, 1.0]$).
3. **Real-World Banking Plumbing:** Models reflect actual payment networks, ISO 8583 response codes, AVS matrices, 3DS 2.x rules, and statutory central bank circulars.
4. **Specification-First Lifecycle:** `spec/` $\to$ Deterministic Invariant Tests $\to$ Implementation $\to$ Raw Data Inspection.

---

---

## License

This project's source code, tests, specifications and original documentation are licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

The research papers under [`docs/papers/`](docs/papers/INDEX.md) are **third-party copyrighted works and are *not* covered by that licence**; they are kept as unlicensed citation copies. The repository is public, so those files are publicly readable; retaining them there is a deliberate decision, not an oversight, and it grants nobody, including this project, a right to redistribute them. The `LICENSE` file states this exception explicitly. If you redistribute this repository, remove `docs/papers/` first.

```
Copyright 2024-2026 the maintainers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```
