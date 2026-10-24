# Prism Generator 7-Day Grounded Realism & Benchmark Roadmap

## Core Philosophy: Anti-Astronaut Grounding & Rigorous Pace
This roadmap defines a disciplined, 7-day iterative plan to transform Prism Generator from a prototype simulator into an empirically grounded, peer-reviewed evaluation environment for fraud detection and explainability.

Every day follows the strict **R-V-P-I-V** methodology:
$$\text{Research} \longrightarrow \text{Verify Research} \longrightarrow \text{Plan} \longrightarrow \text{Verify Plan} \longrightarrow \text{Implement} \longrightarrow \text{Verify Implementation}$$

We advance **one milestone per day**. No skipping ahead, no rushing, and zero ungrounded buzzwords. Every parameter must cite an official regulatory standard, central bank report, or empirical payment survey.

---

## Daily Schedule & Milestone Overview

| Day | Date | Milestone | Primary Objective | Deliverable | Acceptance Gate |
|:---|:---|:---|:---|:---|:---|
| **Day 1** | Sat, Sep 12 | **US Empirical Calibration** | Fed DCPC & FRPS Grounding | Calibrated US Hawkes $\mu_0$ & spend log-normals in `spec/` & `hawkes.py` | 2-sample KS test $p > 0.05$ against Fed daily frequency & ticket size marginals |
| **Day 2** | Sun, Sep 13 | **India Empirical Calibration** | RBI & NPCI Payment Grounding | Calibrated India rail parameters (UPI, RuPay, POS vs CNP) in `spec/05_india_payment_rails.yaml` | Marginal validation against RBI Annual Report & NPCI volume/value shares |
| **Day 3** | Mon, Sep 14 | **Supervision & Latency Layer** | Verification Latency & Ops Queue | `SupervisionEngine` in `stream.py` with investigator capacity & 21d chargeback lag | Labels carry discovery timestamps; unreviewed fraud delayed/unlabelled |
| **Day 4** | Tue, Sep 15 | **Prequential Evaluation Harness** | Temporal Drift & No Random Splits | Time-ordered train/test splits and concept drift schedules in `benchmark.py` | Strictly temporal evaluation $[0, t] \to [t, t+\Delta t]$; recall decay verified |
| **Day 5** | Wed, Sep 16 | **Causal Intervention Benchmark** | Attack Footprint as True Ground Truth | Standardized intervention delta footprint; SCM repositioned as baseline explainer | Precision@k on $\ge 300$ attacks per playbook with 95% confidence intervals |
| **Day 6** | Thu, Sep 17 | **Backbone Replay Mode** | Injection over Real Legitimate Traffic | `BackboneReplayEngine` supporting IEEE-CIS and ULB transaction streams | Clean injection of synthetic attacks over real backbone with zero label leakage |
| **Day 7** | Fri, Sep 18 | **Governance & PRISM Integration** | Dataset Card & External Replication | Clean README (no buzzwords), BAF-style Dataset Card, end-to-end PRISM evaluation | External detector evaluates on export; explanation recovery certified |

---

## Detailed Daily Breakdown

### Day 1: US Empirical Calibration (Federal Reserve DCPC / FRPS)
- **Problem:** Baseline spending schedules and Hawkes base arrival intensities $\mu_0$ were hand-tuned rather than anchored in empirical cardholder data.
- **Empirical Sources:**
  - Federal Reserve Bank of Atlanta: *Diary of Consumer Payment Choice (DCPC)* (annual series).
  - Federal Reserve Board: *Federal Reserve Payments Study (FRPS)* (triennial core data).
- **Key Parameters to Ground:**
  - Average consumer card payment frequency: $1.8$ to $2.7$ transactions per active cardholder per day.
  - Ticket size marginal distributions by MCC category (Grocery, Dining, Retail, Utilities, Travel) modeled via empirical log-normal parameters $(\mu, \sigma)$.
  - Channel shares: Card-Present (62-68%) vs Card-Not-Present (32-38%).
- **Tasks:**
  1. Extract and document exact DCPC/FRPS parameters in `spec/02_human_personas.yaml`.
  2. Calibrate Hawkes baseline parameters in `prism_generator/hawkes.py`.
  3. Write test suite asserting two-sample distribution convergence against Fed reference tables.

---

### Day 2: India Empirical Calibration (Reserve Bank of India & NPCI)
- **Problem:** Indian payment dynamics require accurate representation of RuPay card adoption, domestic CNP 2FA OTP mandates, and contactless limits.
- **Empirical Sources:**
  - Reserve Bank of India (RBI): *Annual Report on Payment and Settlement Systems*.
  - National Payments Corporation of India (NPCI): *Monthly Payment System Statistics*.
  - RBI Master Direction on Card Issuance and Conduct (2022/2023 updates).
- **Key Parameters to Ground:**
  - Debit vs Credit transaction volume and ticket size ratios in India.
  - Contactless POS transactions under ₹5,000 PIN-free ceiling vs mandatory PIN above ₹5,000.
  - E-commerce domestic CNP: 100% mandatory Additional Factor of Authentication (AFA/OTP).
- **Tasks:**
  1. Update `spec/05_india_payment_rails.yaml` with explicit RBI/NPCI statistical citations.
  2. Align `prism_generator/engine.py` and `rails.py` Indian transaction routing.
  3. Validate Indian approval rates and decline distributions against RBI published baselines.

---

### Day 3: Supervision & Verification Latency Layer
- **Problem:** Synthetic simulators naively give instant labels (`is_fraud = 1` at authorization time), whereas real fraud operations operate under delayed discovery and investigator bandwidth constraints (Dal Pozzolo et al., IEEE TNNLS 2018).
- **Operational Model:**
  - **Investigator Daily Budget:** Operations teams can only inspect the top-$K$ highest-risk alerts generated by point-in-time scoring per day.
  - **Rapid Alert Confirmation:** Transactions reviewed by human investigators receive labels within $24-72$ hours.
  - **Chargeback Dispute Maturity:** Unreviewed fraudulent transactions are discovered only when the victim receives their monthly statement and files a dispute ($21-45$ days latency).
  - **Dark / Unreported Fraud:** Low-ticket micro-probing fraud ($< \$10$) has an empirical $8-12\%$ rate of never being reported by cardholders.
- **Tasks:**
  1. Create `SupervisionEngine` in `prism_generator/stream.py`.
  2. Implement alert queues, investigator review budgets, and chargeback lag schedules.
  3. Export transactions with explicit `discovery_timestamp_utc` and `label_source` (`investigator_alert`, `chargeback_dispute`, `unlabelled`).

---

### Day 4: Prequential Time-Ordered Evaluation Harness
- **Problem:** Shuffled cross-validation flatters models and hides seasonal or adversarial temporal degradation.
- **Methodology:**
  - Strict **prequential evaluation** (train on historical window $[t_0, t_1]$, evaluate on subsequent test window $[t_1, t_2]$).
  - **Adversarial Concept Drift Schedule:** Attackers switch targets, proxy subnets, and merchant categories over the simulation timeline.
- **Tasks:**
  1. Refactor `benchmark.py` to support prequential rolling time-window splits.
  2. Measure model performance decay over time as playbooks mutate.
  3. Assert that no test record has a timestamp prior to or equal to any training record.

---

### Day 5: Causal Intervention Benchmark & Explainer Baseline Suite
- **Problem:** Defining "explanation ground truth" via an artificial SCM formula contradicts real-world forensics where ground truth is the attacker's actual physical intervention.
- **Ground Truth Definition:**
  - The physical intervention footprint is the exact dictionary of attributes modified by the adversary on the would-be transaction vector (e.g., $\Delta \text{amount}$, $\Delta \text{channel}$, $\Delta \text{ip\_distance}$, $\Delta \text{asn}$).
- **Tasks:**
  1. Standardize the physical intervention delta tracking across all playbooks in `prism_generator/agents.py`.
  2. Reposition the SCM as one baseline explainer alongside TreeSHAP and Integrated Gradients.
  3. Evaluate explainer Precision@k and Recall@k against the real attack footprint across $\ge 300$ samples per playbook.

---

### Day 6: Real-World Backbone Mode & Zero-Leakage Continuous Audit
- **Problem:** Simulating legitimate traffic from scratch can create subtle synthetic artifacts. Allowing a real backbone mode eliminates this gap.
- **Methodology (The PaySim Recipe):**
  - Replay an established public legitimate transaction backbone (e.g., IEEE-CIS legitimate transactions or ULB).
  - Inject Prism Generator's adaptive cybercrime syndicates and rail verifier on top of the real stream.
- **Tasks:**
  1. Implement `BackboneReplayEngine` in `prism_generator/engine.py`.
  2. Ensure injection maintains bitwise credit limits and ledger balance consistency.
  3. Run the automated leak audit test suite across both synthetic and backbone exports.

---

### Day 7: Governance, Dataset Card & PRISM Integration
- **Problem:** Academic and institutional credibility requires clean terminology, explicit documentation, and independent replication.
- **Tasks:**
  1. Strip speculative buzzwords ("physics-informed hybrid...", "adversarial intent mesh") from all files; use concrete engineering terms (*Discrete-Event Multi-Agent Simulator with Payment Rail Verifiers*).
  2. Cite the April 2026 behavioral fraud benchmark (*arXiv 2604.13125*) in the README to substantiate why DES outclasses deep generative models (GANs/diffusion) on financial sequences.
  3. Author a comprehensive **Dataset Card** following BAF / NeurIPS Datasets & Benchmarks standards.
  4. Run PRISM's detection and explanation pipeline on an exported Prism Generator benchmark feed to certify zero-leakage faithful explanation recovery.

---

## Active Milestone Tracking
The currently active milestone and daily progress is tracked in [`spec/active_slice.yaml`](spec/active_slice.yaml).
