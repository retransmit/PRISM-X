# EMPIRICAL RESEARCH REPORT: Dispute, Chargeback Lifecycle & Regulatory Resolution Mechanics
**Auditor:** Subagent F (Dispute, Chargeback Lifecycle & Regulatory Resolution Specialist)  
**Recipient:** Parent Lead Agent  
**Statutory & Network Authorities:** Visa Core Rules (VCR / VROL), Mastercard Dispute Rules (MasterCom), 12 CFR Part 1026 (Reg Z - Truth in Lending Act), 12 CFR Part 1005 (Reg E - Electronic Fund Transfer Act), Fair Credit Billing Act (15 U.S.C. § 1666), RBI Circular `RBI/2017-18/15` (`DBR.No.Leg.BC.78/09.07.005/2017-18`), and RBI Circular `RBI/2019-20/67` (`DPSS.CO.PD No.629/02.01.014/2019-20`).

---

### EXECUTIVE SUMMARY
This audit provides the foundational empirical and statutory specifications for the post-authorization dispute and resolution layer of Prism Generator. In real-world payment plumbing, fraud losses and financial resolution are governed not at authorization (T=0), but throughout a multi-stage dispute and settlement pipeline spanning T+1 to T+180 days. 

This document details:
1. **Global Dual-Message Dispute Lifecycles** (Visa Claims Resolution vs. Mastercard MasterCom), formal reason codes (Visa 10.4, Mastercard 4837), second presentment standards, Visa Compelling Evidence 3.0 (CE 3.0), and the $500 arbitration fee hurdle.
2. **Indian Regulatory Framework (RBI)**: Mandatory Zero and Limited Liability tiers under `RBI/2017-18/15` (₹5,000 / ₹10,000 / ₹25,000 caps), 10-day provisional credit, 90-day absolute resolution window, and failed transaction penalties under `RBI/2019-20/67` (₹100/day).
3. **Friendly Fraud & First-Party Misuse**: Microeconomics, chargeback multipliers ($3.75 to $4.23 per $1 lost), monitoring program thresholds (VDMP / ECP), and representment win rate distributions (18% to 35%).

---

## 1. GLOBAL DUAL-MESSAGE DISPUTE LIFECYCLE (VISA VCR & MASTERCARD)

### 1.1 Architecture: Visa Claims Resolution (VCR) vs. Mastercard MasterCom
Prior to April 2018, card disputes operated on an archaic litigation model. Visa restructured its network via **Visa Claims Resolution (VCR)** into two distinct dispute streams:

1. **Allocation Workflow (Fraud & Authorization Disputes):**
   - Applies to Visa Dispute Category 10 (Fraud: 10.1 to 10.5).
   - **Automated Network Liability Assignment:** Visa Resolves Online (VROL) interrogates scheme data at ingestion. If the transaction was a Card-Not-Present (CNP) transaction processed without EMV 3D-Secure (3DS) authentication, the system automatically allocates liability to the Acquirer and initiates an immediate financial debit.
   - **Elimination of Second Presentment:** Acquirers *cannot* submit a standard representment under the Allocation path. To contest, the merchant must file a **Pre-Arbitration** response proving valid Compelling Evidence (or CE 3.0 qualifying data).
2. **Collaboration Workflow (Processing Errors & Consumer Disputes):**
   - Applies to Categories 11 (Authorization), 12 (Processing Errors), and 13 (Customer Disputes).
   - Follows the traditional 3-stage dispute cycle: Dispute (First Chargeback) $\to$ Representment (Second Presentment) $\to$ Pre-Arbitration / Arbitration.

**Mastercard MasterCom Architecture:**
- Retains the classical 3-stage model across all chargebacks: Chargeback $\to$ Second Presentment $\to$ Arbitration.
- However, MasterCom Claims Manager automatically enforces rules (e.g., rejecting Reason Code 4837 if the transaction possesses a valid 3DS SecureCode SLI indicating an issuer liability shift).

---

### 1.2 Formal Dispute Reason Codes for Fraud
*   **Visa Reason Code 10.4: Other Fraud  to  Card-Absent Environment:**
    - Filed when a cardholder asserts they neither participated in nor authorized a remote CNP transaction.
    - Statutory Filing Window: Maximum of **120 calendar days** from the transaction settlement date (or delivery date of goods).
*   **Mastercard Reason Code 4837: No Cardholder Authorization:**
    - The direct Mastercard counterpart to Visa 10.4.
    - Statutory Filing Window: Maximum of **120 calendar days** from the central site processing date.

---

### 1.3 Second Presentment Standards & "Compelling Evidence"
To defend against a first chargeback under Collaboration (Mastercard) or Pre-Arbitration under Allocation (Visa), merchants must produce documentation defined under network rules:
*   **Physical Goods:** Carrier-verified proof of delivery (POD) signed by the cardholder, or photo evidence matching the cardholder's verified residential address; AVS full match (`Y` code).
*   **Digital Goods:** Server download logs, timestamped IP addresses, user account credentials matching prior legitimate sessions, and device fingerprint IDs.
*   **Travel & Entertainment (T&E):** Boarding passes, passenger manifests, hotel check-in folio signed by the guest, or keycard access audit trails.
*   **Recurring Transactions / Subscriptions:** Explicit recurring payment agreement, proof that the cardholder previously accessed the service, and documentation that cancellation terms were disclosed in compliance with card brand rules.

---

### 1.4 Visa Compelling Evidence 3.0 (CE 3.0) Specification
Implemented globally in April 2023, CE 3.0 provides an algorithmic, objective standard to defeat "friendly fraud" disputes under Reason Code 10.4 without subjective human review.

#### Mandatory Eligibility Criteria:
1. **The 2-Transaction History Rule:** The merchant must supply data from at least **two (2) settled, undisputed transactions** made on the same PAN.
2. **The 120 to 365 Day Temporal Window:** Both prior transactions must have settled between **120 calendar days and 365 calendar days** prior to the disputed transaction date.
3. **Dispute-Free Invariant:** Neither of the two prior transactions can have had a fraud report (TC40) or dispute filed against them.
4. **Data Matching Requirements (The 2-Factor Anchor):**
   - The merchant must prove that at least **two (2) specific data elements** match across ALL THREE transactions (the disputed transaction + the 2 historical transactions).
   - **Mandatory Anchor Element:** At least ONE of the matching elements MUST be:
     - **IP Address** (Customer IP at checkout), OR
     - **Device ID / Device Fingerprint Hash**.
   - **Second Corroborating Element:** Must be one of:
     - Customer User ID / Account Login, OR
     - Shipping Address (must match street address and postal code).
   *(Note: Combining Device ID and Device Fingerprint does NOT count as two distinct factors).*

#### Operational Impact:
- **Pre-Dispute Deflection via Verifi Order Insight:** When an issuer initiates a dispute inquiry via VROL, CE 3.0 data is returned via API in real-time. If the invariant is met, the dispute is deflected *before* a chargeback is created.
- **Liability Shift Restoration:** If a chargeback was already allocated, submitting valid CE 3.0 evidence shifts 100% of financial liability back to the Issuer. The issuer is contractually prohibited from escalating to formal arbitration.

---

### 1.5 Pre-Arbitration, Arbitration & Financial Tollgates

| Stage | Visa VCR Allocation (Fraud 10.4) | Mastercard MasterCom (Fraud 4837) | Statutory SLA Window | Financial Risk / Fees |
| :--- | :--- | :--- | :--- | :--- |
| **First Chargeback** | Immediate Debit via VROL | First Chargeback via MasterCom | Max 120 calendar days | Disputed Amount + $15 to $35 Chargeback Fee |
| **Representment** | N/A (Direct to Pre-Arb) | Second Presentment | 30 to 45 calendar days | Reversal of debit if compelling evidence accepted |
| **Pre-Arbitration** | Acquirer submits CE 3.0 / Evidence | Issuer files Pre-Arbitration | 30 calendar days | Issuer accepts or rejects evidence within 30 days |
| **Formal Arbitration** | Ruled by Visa Arbitration Committee | Ruled by Mastercard Committee | 10 to 15 calendar days post Pre-Arb | **$500 USD Filing Fee + $500 USD Administrative Fee (Loser Pays All)** |

#### The $500 Arbitration Economic Veto
Formal arbitration carries a non-refundable **$500.00 USD filing fee** plus potential **$500.00 USD review fees** assessed against the losing party.
*   **Algorithmic Consequence:** For any disputed transaction with an amount $V < \$1,000\text{ USD}$, rational merchants and acquirers will abandon pre-arbitration even if legally in the right, because the downside risk of losing ($-\$1,000\text{ fees} - V$) vastly exceeds the recovery value ($V$).
*   *Engine Rule:* Any dispute $\le \$500$ that fails CE 3.0 automated deflection is accepted by the merchant/acquirer as a net loss.

---

## 2. INDIAN REGULATORY DISPUTE FRAMEWORK (RBI)

### 2.1 Customer Limited Liability Tiers: RBI Circular `RBI/2017-18/15`
Directive `DBR.No.Leg.BC.78/09.07.005/2017-18` (*Customer Protection  to  Limiting Liability of Customers in Unauthorised Electronic Banking Transactions*) establishes statutory liability schedules that override scheme-level rules for Indian banks and cardholders:

```
                  [Unauthorized Electronic Transaction Occurs]
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
[Bank Deficiency / Fault]                                   [Third-Party Breach / Fraud]
  - Cardholder Liability: ₹0                                          │
  - Absolute Bank Liability                                           ▼
                                                    [Customer Reporting Timeline]
                                                                      │
                     ┌────────────────────────────────────────────────┼────────────────────────────────┐
                     ▼                                                ▼                                ▼
              Within 3 Working Days                         Within 4 to 7 Working Days         Beyond 7 Working Days
           Cardholder Liability: ₹0                     Capped Statutory Liability:         Bank Board-Approved Policy
                                                          - BSBD/PMJDY Accounts: ₹5,000     (Subject to RBI Ombudsman)
                                                          - Savings / PPI / Cards ≤ ₹5L: ₹10,000
                                                          - Credit Cards > ₹5L / Current: ₹25,000
```

#### Detailed Breakdown of Liability Schedules:
1. **Zero Liability (Section 6):**
   - *Case A (Contributory Fraud / Negligence by Bank):* 100% bank liability, regardless of whether the customer reports it.
   - *Case B (Third-Party Breach):* Deficiency lies neither with the bank nor customer; customer notifies bank within **three (3) working days** of receiving communication. Cardholder liability is strictly **₹0.00 INR**.
2. **Limited Liability (Section 7):**
   - If reporting occurs within **four (4) to seven (7) working days**, customer liability is capped at the transaction value or the following statutory maximums, whichever is lower:
     - **₹5,000 INR:** Basic Savings Bank Deposit (BSBD) / PMJDY accounts.
     - **₹10,000 INR:** All other Savings Bank accounts, Pre-Paid Instruments (PPI / Wallets), Current Accounts, Cash Credit / Overdraft accounts, and **Credit Cards with credit limits up to ₹5,00,000 INR**.
     - **₹25,000 INR:** Current/Cash Credit/Overdraft accounts of MSMEs, corporate accounts, and **Credit Cards with credit limits above ₹5,00,000 INR** (Super-Premium / HNI cards).
3. **Beyond 7 Working Days (Section 8):**
   - Customer liability is determined pursuant to the bank's Board-approved policy, subject to regulatory scrutiny by the RBI Banking Ombudsman.
4. **Customer Negligence Exception:**
   - Where the loss is due to customer negligence (e.g., sharing OTP, NetBanking PIN, or clicking phishing links):
   - Customer bears **100% of the loss** until the unauthorized transaction is reported to the bank.
   - Any transaction occurring *after* reporting is 100% the bank's liability.

#### Mandatory Reversal Timelines & Burden of Proof:
- **10-Day Provisional Credit (Section 9):** The bank must credit the disputed amount to the customer's account within **ten (10) working days** from the reporting date (shadow reversal / provisional credit). The credit must be value-dated to ensure no loss of interest to the customer.
- **90-Day Absolute SLA (Section 10):** A customer complaint must be definitively resolved within **90 calendar days**. If the bank fails to resolve it within 90 days, the customer must be compensated 100% of the transaction amount, regardless of liability.
- **Burden of Proof (Section 12):** The burden of proving customer negligence or customer fault rests entirely on the bank.

---

### 2.2 Harmonisation of TAT & Customer Compensation: RBI Circular `RBI/2019-20/67`
Under Directive `DPSS.CO.PD No.629/02.01.014/2019-20` (*Harmonisation of Turn Around Time (TAT) and customer compensation for failed transactions*):

*   **Prescribed Turn Around Times (TAT):**
    *   *ATMs (Cash not dispensed, account debited):* Proactive reversal within **$T + 5$ calendar days**.
    *   *Card-to-Card Transfer / POS / E-Commerce (Account debited, merchant confirmation not received):* Auto-reversal within **$T + 5$ calendar days**.
    *   *UPI / IMPS (Beneficiary not credited):* Auto-reversal within **$T + 1$ calendar day**.
    *   *Prepaid Payment Instruments (PPI / Wallets):* Auto-reversal within **$T + 1$ calendar day**.
*   **Mandatory Compensation Penalty:**
    *   If the bank fails to auto-reverse funds within the prescribed TAT, the bank must pay compensation of **₹100 INR per calendar day of delay**.
    *   Compensation must be credited to the customer's bank account **automatically** without requiring a customer claim.

---

## 3. FRIENDLY FRAUD & FIRST-PARTY MISUSE RECOVERY

### 3.1 Economics & Chargeback Multipliers
- **Friendly Fraud Prevalence:** Industry surveys (LexisNexis, Chargebacks911) demonstrate that **60% to 75%** of all CNP fraud chargebacks are attributable to first-party misuse or "friendly fraud" (cardholder forgets transaction, family member uses card without permission, or intentional chargeback abuse).
- **The Chargeback Cost Multiplier:** For every \$1.00 USD in fraudulent chargebacks, the true operational cost to a merchant ranges between **\$3.75 and \$4.23 USD**. This includes:
  1. Loss of merchandise / service rendered (\$1.00).
  2. Non-refundable payment processing fees (2.5% to 3.5%).
  3. Acquirer chargeback administrative fees (\$15.00 to \$35.00 flat fee per chargeback).
  4. Dispute management SaaS / manual investigation labor costs (\$10.00 to \$25.00).
  5. Threat of scheme monitoring fines (VDMP / ECP / VAMP).

### 3.2 Representment Win Rates & Contesting Thresholds
- **Baseline Win Rate:** Traditional representment win rates average **18% to 35%** across all retail sectors due to the high subjective evidentiary standard of issuer review.
- **Win Rates under Visa CE 3.0:** When merchants supply valid CE 3.0 data packets matching the 2-prior-undisputed-transaction rule, deflection and representment win rates surge to **60% to 75%**.
- **Merchant Contesting Decision Boundary:**
  $$\text{Expected Value } \mathbb{E}[\text{Contest}] = P_{\text{win}}(V) \cdot V - C_{\text{rep}} - \mathbb{I}(\text{Arbitration}) \cdot (P_{\text{lose}} \cdot \$500)$$
  Where $C_{\text{rep}}$ is the operational cost to represent (\$15 to \$25). For transactions below \$50, contesting produces a negative expected return unless 100% automated via CE 3.0 API webhooks.
