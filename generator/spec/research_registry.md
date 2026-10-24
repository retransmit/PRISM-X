# Prism Generator Existing Research & Grounding Registry

This registry provides a consolidated, exhaustive catalog of all domain research, statutory regulations, empirical surveys, payment network specifications, and academic literature currently implemented in or grounding the **Prism Generator** simulation and benchmark engine.

---

## 1. Statutory & Regulatory Legal Frameworks

### 1.1 United States Framework
| Statute / Regulation | Citation | Regulatory Authority | Implemented Plumbing in Prism Generator | Code / Spec Location |
| :--- | :--- | :--- | :--- | :--- |
| **Regulation E (Electronic Fund Transfers)** | 12 CFR Part 1005 (§ 1005.6, § 1005.18) | Consumer Financial Protection Bureau (CFPB) | Cardholder liability tiers for unauthorized debit/prepaid transactions: \$50 if reported within 2 days, \$500 within 60 days, unlimited after 60 days. Exemption for EBT. | [`spec/01_financial_instruments.yaml`](01_financial_instruments.yaml), [`spec/research_notes/subagent_a_card_products_and_rails.md`](research_notes/subagent_a_card_products_and_rails.md) |
| **Regulation Z (Truth in Lending)** | 12 CFR Part 1026 (§ 1026.12) / FCBA (15 U.S.C. § 1666) | CFPB / Federal Reserve Board | \$50 statutory ceiling on consumer credit liability; \$0 liability if reported before use or under network Zero Liability policies. | [`spec/01_financial_instruments.yaml`](01_financial_instruments.yaml) |
| **FinCEN Prepaid Access Rule** | 31 CFR Part 1022 (§ 1022.380) | Financial Crimes Enforcement Network | General-Purpose Reloadable (GPR) prepaid card balance and cashout caps (\$10,000 maximum balance without formal banking DDA). | [`spec/research_notes/subagent_a_card_products_and_rails.md`](research_notes/subagent_a_card_products_and_rails.md) |
| **SNAP / EBT Regulations** | 7 CFR § 274.8 | USDA Food and Nutrition Service | Electronic Benefit Transfer (EBT) magstripe-only transactions; historical lack of chip/PIN protections leading to benefit drop-day skimming sweeps. | [`spec/research_notes/subagent_a_card_products_and_rails.md`](research_notes/subagent_a_card_products_and_rails.md) |

### 1.2 Indian Statutory & Central Bank Framework
| Mandate / Circular | Citation | Regulatory Authority | Implemented Plumbing in Prism Generator | Code / Spec Location |
| :--- | :--- | :--- | :--- | :--- |
| **Customer Limited Liability in Unauthorized Electronic Transactions** | Circular `RBI/2017-18/15` | Reserve Bank of India (RBI) | Three-tier statutory liability schedule: Zero liability ($\le 3$ days), capped liability (₹5,000 to ₹25,000 for 4 to 7 days), and bank board policy (> 7 days). | [`spec/05_india_payment_rails.yaml`](05_india_payment_rails.yaml), [`prism_generator/rails.py`](../prism_generator/rails.py) |
| **Card-on-File Tokenization (CoFT)** | Circular `RBI/2021-22/96` | Reserve Bank of India (RBI) | Prohibition of merchants storing plain PANs; mandatory network device tokenization (POS Entry Mode `031`). | [`spec/05_india_payment_rails.yaml`](05_india_payment_rails.yaml) |
| **Mandatory Additional Factor of Authentication (AFA)** | Master Direction on Card Issuance & Conduct | Reserve Bank of India (RBI) | Mandatory domestic CNP step-up (OTP/SMS 2FA) on e-commerce; contactless NFC tap-and-pay PIN-free ceiling at ₹5,000 with 5-consecutive-tap chip challenge. | [`spec/05_india_payment_rails.yaml`](05_india_payment_rails.yaml) |
| **Citizen Financial Cyber Fraud Reporting System (CFCFRMS)** | MHA / I4C Helpline 1930 Guidelines | Indian Ministry of Home Affairs (I4C) | "Golden Hour" rapid freeze liens on destination beneficiary mule accounts across cascading payment nodes within 2 to 4 hours of victim report. | [`spec/05_india_payment_rails.yaml`](05_india_payment_rails.yaml) |
| **Rule 114B (High-Value Cash & Jewellery PAN Reporting)** | Income Tax Rules, 1962 (Rule 114B) | Central Board of Direct Taxes (CBDT) | Mandatory PAN reporting for purchase of bullion/jewellery exceeding ₹2,00,000; adversaries split transactions (smurfing) below ₹2L threshold during festivals. | [`spec/05_india_payment_rails.yaml`](05_india_payment_rails.yaml) |

---

## 2. Payment Rail Protocols & Network Operating Rules

### 2.1 ISO Financial Messaging Standards
* **ISO 8583 (Financial transaction card originated messages)**:
  * Message Type Identifiers (MTI): `0100` (Auth Request), `0110` (Auth Response), `0200` (Clearing Presentment), `0210` (Clearing Response), `0420` (Reversal).
  * Field 39 Response Codes: `00` Approved, `05` Do Not Honor, `10` Partial Approval, `14` Invalid Card, `51` Insufficient Funds, `57` Transaction Not Permitted, `59` Suspected Fraud, `63` Security Violation, `65` Activity Limit Exceeded, `82` Invalid CVV.
  * Field 11 (STAN), Field 37 (RRN), Field 38 (Auth Code), Field 22 (POS Entry Mode).
* **ISO 20022**:
  * Rich XML/JSON schema integration for clearing and settlement feeds.

### 2.2 Card Brand Operating Manuals
* **Visa Core Rules (VCR) & Visa Product and Service Rules**:
  * Pre-Authorization Holds: Automated Fuel Dispenser (AFD - MCC 5542) standard hold of \$175.00 USD vs. actual pumped fuel cutoff.
  * Dining (MCC 5812): Acquirer 20% tip tolerance window without secondary incremental auth.
  * Lodging (MCC 7011) & Car Rental (MCC 7512): Folio incidental holds up to 30 days.
  * Visa Claims Resolution (VCR) / Compelling Evidence 3.0 (Visa CE 3.0): Pre-dispute deflection rules using historical unreversed transactions.
  * Visa Account Attack Intelligence (VAAI): Detection rules for distributed PAN Enumeration Attacks (PEA).
  * Stand-In Processing (STIP): Issuer timeout SLA (>2.0s to 2.5s) triggering network switch STIP floor limits (\$100 to \$2,500) and Negative Card File (NCF) checks.
* **Mastercard Transaction Processing Rules (MTPR)**:
  * Dual-message settlement matching; single-message PIN debit processing rules.
  * Excessive Chargeback Program (ECP) thresholds: 1.5% and 100 bps dispute-to-sales ratios.

---

## 3. Empirical Economic & Consumer Surveys

* **Federal Reserve Bank of Atlanta: Diary of Consumer Payment Choice (DCPC)**:
  * Annual diary data providing empirical payment frequencies (1.8 to 2.7 transactions/day per active consumer).
  * 24-hour diurnal circadian arrival schedules: nocturnal suppression (<4.5% between 01:00 and 05:00) with peaks during mid-day retail hours.
  * Persona cohort calibration: 7 verified demographic cohorts across income, age, and channel preferences.
* **Federal Reserve Board: Federal Reserve Payments Study (FRPS)**:
  * Triennial payment data for Card-Present (62 to 68%) vs Card-Not-Present (32 to 38%) distribution splits and general credit/debit ratio baselines.
* **U.S. Bureau of Labor Statistics (BLS) Consumer Expenditure Surveys**:
  * Category-specific ticket size log-normal distributions $(\mu, \sigma)$ across Grocery, Dining, Utilities, Travel, and General Retail.

---

## 4. Adversarial & Cybercrime Forensics

Calibrated to darknet threat intelligence, FBI Internet Crime Complaint Center (IC3) reports, and payment security research:
* **Micro-Auth Card Testing**: Low-dollar (\$0.50 to \$2.00) probes exploiting AVS `Z` (ZIP-only match) bypass on unmonitored charity/donation portals.
* **Account Takeover (ATO) with 14-Day Dormancy**: Silent credential stuffing followed by a 14-day observation period before high-velocity draining.
* **Sleeper Bust-Outs**: Synthetic identity credit line seasoning for 6 to 18 months, followed by rapid multi-card limit exhaustion and ACH float kiting.
* **Apple Pay "Yellow Path" Token Provisioning Fraud**: Intercepting or social engineering one-time activation passcodes to provision victim cards onto burner iPhones.
* **Distributed PAN Enumeration Attacks (PEA / BIN Attacks)**: Multi-threaded algorithmic guessing of expiration dates and CVVs across wide gateway arrays.
* **Triangulation Fraud**: Rogue e-commerce storefronts selling discounted goods, buying from legitimate merchants using stolen cards, and pocketing clean buyer cash.
* **Reverse-Proxy Vishing & APK SMS Stealers**: Real-time MFA bypass (e.g., Evilginx2) and sideloaded Android banking malware capturing OTP streams.

---

## 5. Academic Explainable AI (XAI) & ML Literature

* **Quantus (JMLR 2023)**: *Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations* (Hedström et al.):
  * Standardized XAI evaluation metrics: Kendall's $\tau_b$ rank concordance, Spearman's $\rho$, Precision@k support recovery, and Relative Attribution Error (RAE).
* **OpenXAI (NeurIPS 2022)**: *Towards a Comprehensive Evaluation Benchmark for Explainable AI* (Agarwal et al.):
  * Ground-truth synthetic benchmarks for assessing post-hoc explainers against exact underlying data-generating processes.
* **Verification Latency & Class Imbalance in Fraud (IEEE TNNLS 2018)**: *Credit card fraud detection: a realistic modeling and a novel learning strategy* (Dal Pozzolo et al.):
  * Formal modeling of feedback delays, delayed fraud labels (21 to 45 days chargeback maturity), investigator alert inspection budgets, and prequential time-ordered validation.
* **Bank Account Fraud (BAF) Suite (NeurIPS Datasets & Benchmarks 2022)**: (Jesus et al.):
  * Design principles for realistic fraud datasets, dataset documentation cards, and avoiding temporal leakage.
* **Aumann-Shapley & Owen Multilinear Forms**:
  * Axiomatic game-theoretic feature attribution in continuous and probability spaces via 128-point Gauss-Legendre path integration (Integrated Gradients).
