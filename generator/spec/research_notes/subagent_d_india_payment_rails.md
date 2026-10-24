# Exhaustive Empirical Census: Indian Payment Ecosystem, Regulatory Rails & Cybercrime Mechanics
**Source Authorities:** Reserve Bank of India (RBI) Master Directions & Circulars (PSS Act 2007), National Payments Corporation of India (NPCI) Operating Guidelines, Ministry of Home Affairs (MHA) Indian Cyber Crime Coordination Centre (I4C) Compendia, Telecom Regulatory Authority of India (TRAI) Directives, Income Tax Rules (1962).  
**Auditor:** Subagent D (Indian Payment Ecosystem, Regulatory Rails & Cybercrime Specialist)

---

## 1. Regulatory & Payment Rail Plumbing (RBI & NPCI Directives)

### 1.1 Mandatory Additional Factor of Authentication (AFA)
Governed under Section 10(2) and Section 18 of the Payment and Settlement Systems Act, 2007.
*   **Mandatory Rule:** Mandatory AFA for all domestic online CNP transactions (SMS OTP or 3DS / RuPay PaySecure).
*   **Contactless Tap-and-Pay Limit:** Up to **₹5,000 INR** without PIN (`RBI/2020-21/73`). Transactions $> ₹5,000.01$ strictly force PIN prompt. Consecutive contactless transactions capped (typically 5 transactions / ₹15,000 cumulative velocity before forcing contact chip-and-PIN).
*   **e-Mandates for Recurring Transactions:** Up to **₹15,000 INR** standard; up to **₹1,00,000 INR** for mutual fund SIPs, insurance premiums, and credit card bill payments (`RBI/2023-24/90`). Mandates require 24-hour advance pre-debit SMS/email advice with opt-out link.
*   **Small-Value Offline Payments:** Up to **₹500 INR** per transaction; total offline balance cap of **₹2,000 INR** (`RBI/2021-22/150`). UPI Lite per-tx cap is **₹1,000 INR** (aggregate ₹5,000 INR).

### 1.2 Card-on-File Tokenization (CoFT) Mandate (`RBI/2021-22/96`)
*   With effect from October 1, 2022, no entity other than the card issuer and payment network may store raw card data (16-digit PAN, CVV, expiry). Merchants may only store last 4 digits, issuer name, and network.
*   Surrogate Device-PAN / CoF tokens issued by TSPs (VTS, MDES, NPCI Token Connect) are bound to the specific Merchant ID (MID).

### 1.3 Domestic Network (RuPay) vs. Global Schemes
*   **RuPay on UPI:** Links RuPay credit cards directly to UPI VPAs for P2M QR scan-and-pay. 0% MDR for merchants with turnover $\le ₹20\text{L}$ on transactions $\le ₹2,000\text{ INR}$. P2P credit transfers strictly barred.
*   **Data Localization (`RBI/2017-18/153`):** 100% of end-to-end payment system data must be stored exclusively in India. Foreign processing data must be purged overseas and stored in India within 24 hours.

### 1.4 The "International Non-3DS Route-Around" Loophole
*   **RBI Out-of-the-Box Card Controls (`RBI/2019-20/142`):** All new cards issued with Domestic CNP, International, and Contactless disabled by default.
*   **The Exploit:** When an Indian cardholder enables "International CNP" for overseas travel/SaaS, attackers route compromised PAN/CVV through foreign merchants (US, EU, Singapore) that do not support 3DS. Because the merchant cannot invoke 3DS, the Indian bank's host either declines all foreign commerce or approves without OTP. Attacks are timed between 01:30 AM and 05:30 AM IST while the cardholder is asleep.

---

## 2. Indian Card Product Taxonomy

| Card Category | Representative Products | Typical Limit / Balance Range (INR ₹) | Regulatory & Financial Mechanics | Target Fraud Risks |
| :--- | :--- | :--- | :--- | :--- |
| **Kisan Credit Card (KCC)** | SBI KCC, PNB Kisan, NABARD RuPay | ₹25,000 to ₹3,00,000 (up to ₹10,00,000) | Crop-season Kharif/Rabi cycle; 4% net APR (with prompt repayment incentive); annual rollover | BC/kiosk skimming, fake land revenue identity theft |
| **PMJDY RuPay Debit** | SBI, BoB Jan Dhan RuPay Debit | ₹0 to ₹10,000 DDA; ₹10,000 overdraft | Financial inclusion BSBDA rules; ₹2L accidental insurance (requires 1 tx in 90 days) | AePS/Micro-ATM silicone thumb spoofing; cyber money mule recruitment |
| **Entry / FD-Backed Credit** | OneCard, IDFC WOW, Kotak 811 | ₹5,000 to ₹50,000 (80%-100% of FD lien) | 100% lien against fixed deposit; revolving APR 24%-42%; no CIBIL score required | Micro-auth testing; rent portal liquidation before lien enforcement |
| **Mid-Tier Salaried Rewards** | HDFC Regalia/Millennia, ICICI Coral, Axis Flipkart | ₹50,000 to ₹5,00,000 (Median: ₹1,50,000) | Revolving APR 42%-45% p.a.; high No-Cost EMI penetration; 45-50 day credit cycle | Fake reward points vishing, APK SMS forwarders, rent portal cashouts |
| **Super-Premium / HNI** | HDFC Infinia Metal, Axis Magnus, ICICI Emeralde | ₹10,00,000 to ₹50,00,000+ (Dynamic line) | 1.5%-2.0% forex markup; 3.3%-9.9% reward yields; 1:1 airline mile transfers | Concierge ATO, nocturnal international non-3DS bypass, loyalty miles theft |
| **Commercial & Corporate** | HDFC Corporate Platinum, Axis Purchasing | ₹5,00,000 to ₹50,00,000+ pooled | Corporate liability; 30-day settlement; Level II/III data | Vendor portal compromise, fake GST invoice routing via payment aggregators |

---

## 3. Indian Consumer Behavioral Patterns & Quirks

1.  **Salary Cycles & Repayment Surges:**
    *   Private sector salaries credit 28th to last day; government salaries credit 1st to 3rd.
    *   Bill payment platforms (CRED, CheQ, NetBanking BBPS) experience a 3.8x to 5.2x volume surge between the 5th and 10th of the month.
    *   Liquidity squeeze between the 20th and 28th shifts retail purchases from UPI/debit to credit cards.
2.  **The "No-Cost EMI" Architecture:**
    *   RBI bans true 0% loans (`RBI/2013-14/292`); "No-Cost EMI" operates via an upfront merchant discount equal to the interest charge.
    *   Non-refundable: 18% GST on interest + one-time processing fee (₹99 to ₹199 + 18% GST).
3.  **Festival Spikes & Dhanteras Gold Splitting:**
    *   Q3/Q4 festive sales (Flipkart BBD / Amazon GIF) generate a 400% to 800% velocity surge over baseline.
    *   Dhanteras / Akshaya Tritiya: Under Rule 114B of the Income Tax Rules 1962, quoting PAN is mandatory for transactions $> ₹2,00,000\text{ INR}$. Cardholders routinely split jewellery purchases into multiple swipes (e.g., ₹1,85,000 and ₹1,65,000 spaced 90 seconds apart) to stay below the reporting threshold.
4.  **Fuel Surcharge Waiver Mechanics (MCC 5541):**
    *   1.0% surcharge levied at pumps; waived for transactions between ₹400 and ₹4,000 INR (capped at ₹250 to ₹500/month).
    *   18% GST on surcharge is non-refundable by law.
5.  **Cards vs. UPI Wallet Split:**
    *   UPI dominates micro-payments ($>75\%$ of UPI tx are $<₹500\text{ INR}$).
    *   Cards reserved for purchases $>₹2,000\text{ INR}$, travel, electronics, dining, and e-commerce EMIs.

---

## 4. Indian Adversarial Attack Playbooks

*   **Playbook IN-1: Reverse-Proxy Vishing & "Digital Arrest":** High-urgency impersonation (electricity bill disconnection, CBI/cyber police video calls) paired with real-time reverse proxy phishing kits (Modlishka/Evilginx2) to intercept the 6-digit OTP within its 180-second validity window.
*   **Playbook IN-2: Malicious Android APKs (SMS/Screen Stealers):** Side-loaded APKs via WhatsApp (`PM_Kisan.apk`, `Electricity_Bill.apk`) abusing `AccessibilityService` and `RECEIVE_SMS` to silently exfiltrate OTPs directly to attacker Telegram bot APIs while suppressing the incoming SMS chime.
*   **Playbook IN-3: International Non-3DS Bypass:** Routing compromised PAN/CVV through foreign merchants lacking 3DS/SCA between 01:30 and 05:30 AM IST.
*   **Playbook IN-4: Credit-to-Bank Rent Portal Liquidation:** Using rent/tuition payment apps (Housing, NoBroker, Magicbricks) to liquidate credit lines to Layer 1 money mule bank accounts via IMPS for a 1.2% to 2.0% fee, followed by instant P2P USDT crypto conversion.
*   **Playbook IN-5: SIM Swap / eSIM Hijacking:** Bypassing telecom 24-hour SMS blackout via Friday evening timing, IVR voice OTPs, or social engineering.
*   **Defense Protocol (CFCFRMS / 1930):** Citizen Financial Cyber Fraud Management System multi-bank automated lien marking across mule account layers within the "Golden Hour".
