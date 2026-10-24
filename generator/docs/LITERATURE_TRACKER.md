# Prism Generator: Slide-by-Slide Research Literature & Citation Tracker

This document tracks all academic papers, central bank research, and industry benchmark studies **strictly mapped slide by slide** to match your presentation flow.

Every claim made on every slide is tied to its author, year, publication venue and, where one exists, a local copy in [`papers/`](papers/INDEX.md).

**Quotation policy:** text marked *Key point (paraphrase)* is a paraphrase, not a sentence lifted from the paper. No verbatim quotation is presented here unless it is labelled *verified verbatim* and reproduced from a local copy. Local copies are linked with repository-relative paths; a citation with no link means this repository holds no copy of that paper.

---

## Slide 1: Title Slide
* **Slide Title:** Prism Generator: Explainable AI for Payment Fraud Detection
* **Foundational Standard:** 
  * **ISO 8583 / ISO 20022:** Financial transaction card originated messages ,  Interchange message specifications.
  * **Federal Reserve Payments Study (FRPS):** Triennial core payment benchmarks.

---

## Slide 2: Content (Table of Contents)
* Agenda and presentation structural roadmap.

---

## Slide 3: Introduction (The Need for Explainable AI)

### Pillar 1: Customer Impact (High False Alarms & Churn)
* **Citation:** Javelin Strategy & Research (2015 / 2023 updates), *"The False-Positive Problem"*.
  * **Key Data Point:** False-positive card declines cost U.S. merchants **\$118 Billion annually** (~13x the cost of actual fraud). 32% of customers abandon the merchant and 26% stop using that card after an unexplained decline.
* **Citation:** Nobel et al. (2024), *"Unmasking Banking Fraud: Unleashing the Power of Machine Learning and Explainable AI on Imbalanced Data"*, *Information (MDPI)*.
  * **Exact Finding:** Demonstrates that high class imbalance in card payments causes black-box models to over-flag benign spending anomalies, proving that reducing false positives requires transparent local feature attributions.
* **Citation:** Faruk et al. (2025), *"Explainable AI (XAI) for Fraud Detection: Building Trust and Transparency in AI-Driven Financial Security Systems"*, *TechRxiv / IEEE*.
  * **Exact Finding:** Models optimizing solely for anomaly detection without explainability layers generate unacceptable false-alarm spikes that alienate legitimate cardholders.

### Pillar 2: Analyst Impact (Alert Fatigue & Investigator Bottleneck)
* **Citation:** Andrea Dal Pozzolo et al. (2018), *"Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy"*, *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)*.
  * **Local PDF:** `docs/papers/1707.02640.pdf` _(no local copy: the file previously stored here contained a different paper and was removed; see [INDEX](papers/INDEX.md))_
  * **Key point (paraphrase):** frames card-fraud detection as a realistic, drift-prone problem in which labels arrive late and verification is a bounded, sequential investigation task. A sentence previously presented here as an exact quotation could not be reproduced from any copy available to this repository and has been removed; quote this paper only from the publisher's PDF.
* **Citation:** Systematic Review (2024), *"Explainable artificial intelligence (XAI) in finance: a systematic literature review"*, *Artificial Intelligence Review (Springer)*.
  * **Key point (paraphrase):** surveys how explanation methods are used in finance and argues that analysts need more than a scalar risk score to act on an alert. Presented as a paraphrase: no local copy of this review is held here, so no verbatim wording is asserted.
* **Citation:** Aljunaid et al. (2025), *"Secure and Transparent Banking: Explainable AI-Driven Model for Financial Fraud Detection"*, *Journal of Risk and Financial Management*.
  * **Exact Finding:** Probability scores alone cannot support fraud operations; human investigators require decomposed attribution values to legally substantiate card freezes.

### Pillar 3: Model Impact (Trust, Shortcuts & Ground-Truth Dilemma)
* **Citation:** Hedström et al. (2023), *"Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations"*, *Journal of Machine Learning Research (JMLR)*.
  * **Local PDF:** [`docs/papers/2202.06861.pdf`](papers/2023_hedstrom_quantus_responsible_evaluation_xai.pdf)
  * **Key point (paraphrase):** Quantus supplies reference-based metrics (faithfulness, robustness, localisation) for scoring explanation methods, which is why this project scores explainers against the simulator's known intervention effects rather than against the explainer's own output. Paraphrase; the paper is held locally if a verbatim sentence is needed.
* **Citation:** Cynthia Rudin (2019), *"Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead"*, *Nature Machine Intelligence*.
  * **Local PDF:** [`docs/papers/1811.10154.pdf`](papers/2019_rudin_stop_explaining_black_box_high_stakes.pdf)
  * **Exact Finding:** Demonstrates that post-hoc explanations for black-box models are often unfaithful and fail to represent what the model is actually computing in high-stakes domains.

---

## Slide 4: Literature Review (Comparative Literature Survey)

| Domain | Key Papers & Authors | Venue & Year | Local PDF Path | Core Takeaway / Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **XAI Feature Selection & Labeling** | Walauskis & Khoshgoftaar | *IEEE Access 2025* | [`docs/papers/walauskis_khoshgoftaar_ieee_access_2025.pdf`](papers/2025_walauskis_shap_feature_selection_unsupervised_fraud_labeling.pdf) | Used SHAP feature ranking to guide unsupervised labeling on Kaggle fraud data; tested on masked PCA components without ground-truth attribution verification. |
| **Traditional ML in Fraud** | Dal Pozzolo et al. | *IEEE TNNLS 2018* | `docs/papers/1707.02640.pdf` _(no local copy: the file previously stored here contained a different paper and was removed; see [INDEX](papers/INDEX.md))_ | Modeled concept drift and verification latency, but evaluated on masked PCA data. |
| **Post-Hoc Tree Explainers** | Lundberg & Lee (TreeSHAP) | *NeurIPS 2017* | [`docs/papers/1705.07874.pdf`](papers/2017_lundberg_unified_approach_interpreting_model_predictions_shap.pdf) | Fast exact Shapley values for trees, but assumes feature independence causing correlation leakage. |
| **Limitations of Shapley Values** | Kumar et al. | *ICML 2020* | [`docs/papers/2002.11097.pdf`](papers/2020_kumar_problems_with_shapley_value_feature_importance.pdf) | Proves mathematically that Shapley values can assign positive attribution to completely irrelevant correlated features. |
| **Synthetic Financial Data Survey** | "New Money" Systematic Review | *ACM / arXiv 2025* | [`docs/papers/2510.15096.pdf`](papers/2025_finance_new_money_a_systematic_review_of_sy.pdf) | Comprehensive survey showing that deep generative models (GANs/diffusion) fail to preserve accounting invariants. |
| **Causal Graph Fraud Detection** | CaT-GNN | *arXiv cs.LG 2024* | [`docs/papers/2402.14708.pdf`](papers/2024_duan_cat_gnn_causal_temporal_graph_fraud_detection.pdf) | Highlights the necessity of causal graphs over transaction sequences to avoid temporal shortcuts. |

---

## Slide 5: Existing System & Research Gaps

### Gap 1: Anonymized PCA Benchmarks (Kaggle ULB / Dal Pozzolo 2015)
* **Limitation:** Features $V_1 \dots V_{28}$ are anonymized principal components. Explaining "Principal Component 14" provides zero domain meaning for bank adverse action compliance.

### Gap 2: Outdated Simulators (PaySim & BankSim 2016)
* **Citation:** Lopez-Rojas et al. (2016), *"PaySim: A financial mobile money simulator for fraud detection"*.
* **Limitation:** 10-year-old flat table, only 5 primitive transaction types (`CASH_IN`, `TRANSFER`), no card rails, no credit limits, no 3DS, no adaptive cybercrime.

### Gap 3: Statistical Bayesian Simulators (Federal Reserve CardSim 2025)
* **Citation:** Jeffrey S. Allen (Feb 2025), *"CardSim: A Payment Card Transaction Simulator"*, *Federal Reserve Board FEDS 2025-010*.
* **Limitation:** Generates statistical payment curves, but lacks real-time network protocol messaging (ISO 8583), multi-agent closed-loop adaptation, and causal xAI benchmarks.

---

## Slide 6: Problem Statement (The Ground-Truth Evaluation Crisis)
* **Citation:** Agarwal et al. (2022), *"OpenXAI: Towards a Comprehensive Evaluation Benchmark for Explainable AI"*, *NeurIPS 2022*.
  * **Core Problem:** Machine learning explainers are deployed without ground-truth verification. In real transaction logs, nobody knows the true cause ($\Delta \mathbf{x}$), making it impossible to evaluate if SHAP is faithful or hallucinating.
* **Our Problem Formulation:** Formulating fraud explainability as a **causal intervention recovery task**:
  $$\Delta \mathbf{x} = \mathbf{x}_{\text{attack}} - \mathbf{x}_{\text{baseline}}$$

---

## Slide 7: Feasibility Study & Project Scope
* **Legal & Regulatory Feasibility:**
  * **US Regulation E (12 CFR Part 1005) & Regulation Z (12 CFR Part 1026):** Consumer liability limits and adverse action requirements.
  * **Reserve Bank of India (RBI/2017-18/15):** Three-tier statutory customer liability schedule and mandatory OTP (AFA) on domestic CNP transactions.
* **Technical Feasibility:**
  * Discrete-Event Simulation with a 64-bit microsecond monotonic clock measured at ~2,600 events/second (1,414 transactions/second) over a 100,000-transaction run on the development host (2,015-2,626 events/second observed across configurations), reproducible with `python scripts/measure_throughput.py --transactions 100000 --days 60 --region US --seed 42`.

---

## Slide 8: Proposed System (The Prism Generator Causal Architecture)
* **Causal Formulation Citation:** Judea Pearl (2009), *"Causality: Models, Reasoning, and Inference"*, Cambridge University Press.
  * 3-Step Structural Counterfactual Engine: Abduction (latent baseline profile) $\to$ Action (adversarial intervention $\text{do}(A)$) $\to$ Prediction (counterfactual delta).
* **Game-Theoretic Path Integration:**
  * **Sundararajan et al. (2017):** *"Axiomatic Attribution for Deep Networks"* ,  [`docs/papers/1703.01365.pdf`](papers/2017_pre_axiomatic_attribution_for_deep_netw.pdf) (128-point Gauss-Legendre Aumann-Shapley integration).

---

## Slide 9: Methodology ,  Block Diagram & 4-Hop Banking Switch
* **Circadian & Temporal Modeling:**
  * **Hawkes, A. G. (1971):** *"Spectra of some self-exciting point processes"*.
  * **Lewis & Shedler (1979):** Non-homogeneous Poisson Process (NHPP) thinning for diurnal human circadian sleep cycles ($<4.5\%$ nocturnal spend).
  * **Federal Reserve Diary of Consumer Payment Choice (DCPC):** Annual consumer transaction arrival diaries.
* **4-Hop Banking Protocol:**
  $$\text{Payment Gateway} \longrightarrow \text{Visa VAAI (Anti-Enumeration)} \longrightarrow \text{EMV 3DS 2.x ACS} \longrightarrow \text{Issuer Host Authorization (ISO 8583)}$$

---

## Slide 10: Methodology ,  Implementation, Algorithms & Flowchart
* **Point-in-Time Streaming Ledger:**
  * **Welford (1962):** Online one-pass mean and variance calculation without future lookahead bias.
* **Kinematic Travel Limits:**
  * Great-circle Haversine distance assertions ensuring card-present velocities strictly adhere to commercial transport limits ($<900$ km/h).

---

## Slide 11: Methodology ,  Dataset & 4 Partitioned Feeds
* **Institutional Realism Citation:** Sérgio Jesus et al. (2022), *"Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation"*, *NeurIPS Datasets & Benchmarks Track*.
  * **Local PDF:** [`docs/papers/2211.13358.pdf`](papers/2022_jesus_turning_tables_biased_imbalanced_tabular.pdf)
* **Partitioned Institutional Architecture:**
  1. `auth_stream.csv` (ISO 8583 MTI 0100/0110 real-time authorization)
  2. `gateway_telemetry.csv` (Device canvas hashes, ASN types, IP distance)
  3. `clearing_settlement.csv` (Dual-message MTI 0200 clearing with 24 to 72h delay)
  4. `dispute_recovery.csv` (Chargebacks, Visa CE 3.0 deflection, RBI liability tiers)

---

## Slide 12: Results, Empirical Audit & Discussion
* **Empirical Realness Forensic Audit (`scripts/audit_fraud_realness.py`):**
  * **Benford's Law Conformity:** Legitimate spend first-digit $\text{MAD} = 0.0076$ (Threshold $<0.012$).
  * **Nocturnal Trough:** Legitimate night spend $2.77\%$ (US) and $3.13\%$ (IN), strictly conforming to central bank baselines ($<4.5\%$).
  * **Kinematic Violations:** $0$ card-present transactions exceed $900$ km/h.
  * **Realistic Machine Learning Separability:** Logistic Regression achieves $\text{PR-AUC} = 0.6686$ (US) and $0.4994$ (IN) with $17.6\%$ Precision and $82.6\%$ Recall.
* **Quantus / OpenXAI Benchmark Findings (`prism_generator/benchmark.py`):**
  * **Top-3 Support Recovery (Precision@3):** $62.8\%$ (TreeSHAP recovers the primary fraud lever ~6 out of 10 times).
  * **Kendall's $\tau_b$ Rank Concordance:** $0.4439$ (Significant ranking divergence due to correlated financial features).

---

## Slide 13: Conclusion & Future Scope
* **Conclusion:**
  * Proves that current post-hoc explainers (TreeSHAP) exhibit significant ranking divergence and magnitude error when uncalibrated on financial data.
* **Future Scope:**
  * Ingesting external public streams (**IEEE-CIS 2019**, **Feedzai BAF 2022**).
  * Developing interactive triage dashboards for frontline fraud ops investigators.
  * Counterfactual recourse recommendation algorithms.

---

## Slide 14: Formal Bibliography / References (IEEE Style)
1. A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, "Credit card fraud detection: a realistic modeling and a novel learning strategy," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 29, no. 8, pp. 3784 to 3797, 2018.
2. A. Hedström, L. Weber, D. Bareeva, et al., "Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations," *J. Mach. Learn. Res.*, vol. 24, pp. 1 to 11, 2023.
3. S. Jesus, J. Pombal, M. Alves, et al., "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation," in *Proc. NeurIPS Track on Datasets and Benchmarks*, 2022.
4. J. S. Allen, "CardSim: A Payment Card Transaction Simulator," *Finance and Economics Discussion Series 2025-010*, Board of Governors of the Federal Reserve System, 2025.
5. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765 to 4774.
6. C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nat. Mach. Intell.*, vol. 1, no. 5, pp. 206 to 215, 2019.
7. M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic attribution for deep networks," in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 3319 to 3328.
8. Javelin Strategy & Research, "The False-Positive Problem: How Unnecessary Declines Harm Merchants and Disappoint Customers," Javelin Strategy, Tech. Rep., 2015.
