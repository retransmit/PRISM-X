# Exhaustive Empirical Census: Payment Card Product Taxonomy & Payment Rail Plumbing
**Source:** Visa Core Rules (VCR), Mastercard Transaction Processing Rules (MTPR), 12 CFR Part 1005 (Reg E), 12 CFR Part 1026 (Reg Z), Fair Credit Billing Act (15 U.S.C. § 1666), 31 CFR Part 1022 (FinCEN Prepaid Access), PSD2 RTS (Commission Delegated Regulation (EU) 2018/389), EMVCo Tokenization Specifications.
**Auditor:** Subagent A (Payment Card Product & Network Rail Census Specialist)

---

## 1. Complete Product Taxonomy & Operational Parameters

| Instrument Category | Product Codes / Issuers | Typical Credit / Balance Range | Regulatory Framework | Statutory Cardholder Liability Cap | Default Velocity / Daily Cap | Primary Adversarial Attack Vector |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Subprime / Secured Credit** | Visa Classic (F), Credit One, OpenSky | Secured: $200 to $1,000; Unsecured: $300 to $750 (fee-loaded to $150 to $225 open) | Reg Z (12 CFR § 1026) / CARD Act 2009 | $50 (Zero Liability applies) | 3 to 5 tx/day; $250 to $500/day; ACH refresh 3 to 7 days | Synthetic identity bust-out; low-dollar card testing |
| **Classic Consumer Credit** | Visa Traditional (A), MC Standard (MCC) | $1,000 to $5,000 (median: $2,500) | Reg Z / FCBA | $50 (Zero Liability applies) | 8 to 12 tx/day; $1,500 to $3,000/day; Cash adv cap 20-30% | Bulk CNP e-commerce credential dumps ($400-$900 tickets) |
| **Rewards / Signature Credit** | Visa Signature (C), MC World (MCW) | $5,000 to $25,000 (median: $12,500) | Reg Z / FCBA | $0 Zero Liability | 15 to 25 tx/day; $5,000 to $10,000/day; wide geo-velocity | Account Takeover (ATO), SIM swap, loyalty points draining |
| **Ultra-Premium Credit** | Visa Infinite (I), MC World Elite, Amex Plat/Centurion | $20,000 to $100,000+ (Amex NPSL dynamic shadow limit) | Reg Z / FCBA | $0 Zero Liability (VIP concierge) | 30 to 50+ tx/day; $25k to $100k/day; Single tx up to $50k+ | Dormant line waking; exploitation of VIP false-decline fear |
| **Traditional Checking Debit** | Visa Debit (F/D), MC Debit, Interlink, Pulse | DDA checking: $200 to $15,000 (median: ~$3,000) | Reg E (12 CFR Part 1005) / EFTA | Strict: $50 (<=2d), $500 (<=60d), Unlimited (>60d) | ATM: $500 to $1,000/day; POS: $2,000 to $5,000/day; 15-20 tx/day | Magstripe skimming + PIN harvest; rapid ATM cashout |
| **PIN Debit Rails** | Interlink, STAR, NYCE, Pulse, Maestro | DDA checking | Reg E / ANSI X9.24 DUKPT | Extremely narrow chargeback if PIN validated | POS / ATM Limits | PIN brute-force at compromised unattended terminals |
| **GPR Prepaid Cards** | Visa Prepaid (P), Green Dot, Netspend | $100 to $10,000 (FinCEN 31 CFR § 1022 cap) | Reg E (§ 1005.18 Prepaid Rule) | $50 / $500 / Unlimited | POS: $2,500 to $5,000/day; ATM: $500 to $1,000/day | Money mule liquidation; bypass AVS |
| **Payroll Cards** | Visa/MC Payroll | Direct deposit wages: $200 to $4,000 | 12 CFR § 1005.18 (Mandatory use banned) | $50 / $500 / Unlimited | Standard debit velocity | Employer payroll portal phishing (ADP, Workday diversion) |
| **Government EBT (SNAP/TANF)** | Quest Network | SNAP: $200 to $900/mo; TANF: Cash | USDA 7 CFR § 274.8 (Exempt from Reg E) | Zero statutory protection (pre-2023) | Balance exhaustion | Mass POS skimming on 1st-10th benefit drop; no chip |
| **Direct Express** | Comerica federal contractor | Social Security / VA: $1,000 to $2,500/mo | 31 CFR Part 208 / 12 CFR § 1005.15 | Reg E protections apply | Standard debit velocity | Phone social engineering of elderly benefit recipients |
| **Small Business Credit** | Visa Business (G), Amex Blue Business | $5,000 to $50,000 (personal guarantee) | Commercial contract (Exempt Reg Z if >10 cards) | Full corporate liability | Credit limit daily | Compromised ad account spend (Meta/Google Ads) |
| **Corporate T&E Cards** | Visa Corporate (K), MC Corporate | $10,000 to $100,000+ pooled lines | Commercial contract | Full corporate liability | Restricted to travel MCCs (Airlines, Hotels, Dining) | Booking tool compromise (Concur); inflated phantom folios |
| **Purchasing Cards (P-Cards)** | Visa Purchasing (S), MC Purchasing | $25,000 to $500,000+ monthly | Commercial contract / Level II & III | Full corporate liability | Strict single-tx limit (e.g. $2,500) & MCC whitelist | Business Email Compromise (BEC) supplier invoice diversion |
| **Commercial Fleet Cards** | WEX, Voyager, Comdata, Fuelman | $500 to $2,500/vehicle | Commercial contract | Full corporate liability | Restricted to fuel (5541/5542) & repair (7538) | Driver ID & odometer bypass; ghost fueling into bladders |
| **HSA / FSA Cards** | Visa Healthcare (N), SIGIS IIAS | $100 to $8,000 (statutory annual caps) | IRS Rev. Rul. 2003-43 / SIGIS IIAS | Reg E / Reg Z | Daily available balance | Rogue medical MCC spoofing; retail return-to-cash loops |
| **Private-Label Store Cards** | Synchrony, Bread, Citi Retail | $300 to $3,000 (APR 29.99% to 34.99%) | Reg Z / Open-end consumer | $50 (Zero Liability applies) | Restricted to parent retailer brand | In-store register account lookup takeover with fake DL |
| **BNPL Single-Use Virtual** | Klarna, Affirm, Zip | $50 to $1,500 dynamic; 24-48h TTL | Reg Z (CFPB 2024 Rule) | $50 (Zero Liability applies) | 100-115% over-auth buffer; single-merchant lock | Burner identity stacking across 4 apps simultaneously |
| **Tokenized DPAN (Apple Pay)** | VTS / MDES Token Vault | Mirrors FPAN credit line | Mirrors FPAN | $0 (Biometric liability shift) | Mirrors FPAN | Token provisioning fraud: OTP social engineering (Yellow path) |
| **Merchant-Locked Virtual** | Privacy.com, Ramp | User-defined ($10 to $10,000) | Mirrors funding DDA/Line | $0 | Bound to first authorized Acquirer MID/MCC | Platform master API credential stuffing |

---

## 2. Payment Rail Plumbing & Protocol Mechanics

### 2.1 Dual-Message System (DMS) vs Single-Message System (SMS)
*   **Dual-Message (Visa Base I/II, Mastercard Banknet/IPM):**
    *   *Real-time (T=0):* 0100 Auth Request $\to$ 0110 Auth Response (`00` Approved, `05` Do Not Honor, `51` Insufficient Funds). Places pending memo-hold on cardholder ledger.
    *   *Clearing & Presentment (T+1 to T+3 business days):* Merchant batch close generates Base II TC05 or Mastercard IPM clearing files matching original auth via RRN (DE 37), STAN (DE 11), Auth Code (DE 38), and TID.
    *   *Hold Expiry:* Auto-drops at 7 calendar days (retail) or 30 days (lodging/car rental). Late clearing posts anyway, causing unexpected overdrafts.
*   **Single-Message (EFT PIN Debit / ATM):**
    *   0200 Financial Transaction Request Inline. Real-time ledger debit at T=0. Zero hold lag.

### 2.2 Pre-Authorization Holds & Clearing Variance
*   **Automated Fuel Dispensers (AFD - MCC 5542):**
    *   Modern standard hold: **$175.00 USD** (fleet: $350.00).
    *   Mandatory partial authorization (Field 39 = `10`): If card has only $42 available, approves $42, and dispenser motor halts pump at delivered fuel value of $42.00. Final presentment clears exact pumped volume.
*   **Lodging (MCC 7011) & Car Rental (MCC 7512):**
    *   Hold formula: $(\text{Room} + \text{Tax}) \times \text{Nights} + (\$50\text{--}\$100 \times \text{Nights})$. Hold persists up to 30 calendar days.
*   **Dining & Restaurants (MCC 5812 / 5813):**
    *   **20% Tip Tolerance Window:** Acquirers may clear up to 120% of authorized check without secondary authorization. If settlement exceeds 120% without incremental auth, cardholder bank has immediate chargeback rights (Visa Reason 13.1 / Mastercard 4837).

### 2.3 Stand-In Processing (STIP)
*   **Trigger:** Issuer host timeout (>2.000 to 2.500 seconds SLA) or link failure. Network switch immediately engages STIP.
*   **STIP Floor Limits:** Standard: $100 to $250; Gold/Signature: $500 to $1,000; Ultra-Premium/Corporate: $2,500.
*   **Controls:** Negative Card File (NCF) exception hotlist check, Positive Balance File (PBF) check, max 2 to 3 consecutive offline approvals per outage window. 4th consecutive attempt is hard-declined (`05`).
*   **Signaling:** Field 39 = `00`, Field 44.5 (Visa) or Field 48.22 (Mastercard) indicates network STIP approval. Issuer is contractually liable.

### 2.4 Address Verification Service (AVS) Response Matrix
*   Extracts numeric characters from street address + 5/9 digit postal code:
    *   `Y`: Full match (street & 5-digit zip).
    *   `A`: Street matches, zip does not.
    *   `Z`: 5-digit zip matches, street does not (Primary bypass for gas pumps & digital micro-merchants using whitepages lookups).
    *   `N`: Neither matches (High fraud indicator).
    *   `U` / `G`: International issuer non-participating (Merchant dilemma: decline 80% legitimate foreign buyers or accept 100% fraud chargeback liability).

### 2.5 EMV 3-D Secure 2.x & PSD2 RTS Exemptions (Regulation (EU) 2018/389)
*   **Article 16 (Low Value):** $\le €30$ EUR remote transactions. Reset circuit-breaker: SCA mandated if cumulative non-SCA spend $>€100$ EUR or consecutive non-SCA count $>5$ transactions.
*   **Article 18 (Transaction Risk Analysis - TRA):**
    *   $\le €100$ EUR: PSP fraud rate $\le 0.13\%$ (13 bps).
    *   $€100.01\text{--}€250$ EUR: PSP fraud rate $\le 0.06\%$ (6 bps).
    *   $€250.01\text{--}€500$ EUR: PSP fraud rate $\le 0.01\%$ (1 bp).
    *   $>€500$ EUR: **Strictly forbidden.** SCA is legally mandatory.
*   **Article 13 (Trusted Beneficiary / Whitelisting):** Exempt once cardholder adds merchant during SCA challenge.
*   **Article 14 (Recurring / Subscriptions):** Fixed-amount series exempt after initial CIT setup.
