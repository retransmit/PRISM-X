# Public Indian data on payment fraud: what exists, what it says, and what we build on it

**Date:** 23 September 2026. **Purpose:** the evidence base for the Indian transaction generator and for the fraud-detection track of Prism. Every figure below was read from the source named beside it on the date above; nothing is taken from a secondary dataset site. Where two official series disagree, both are shown and the reason is stated.

## 1. The short version

1. **No public Indian transaction-level payment dataset exists.** Not from RBI, NPCI, a bank, or a government portal. Every "Indian transactions" file on data-sharing sites is synthetic, made by an individual, with no stated generating process. Real transaction data lives inside banks, NPCI and the RBI's fraud registry, and the law and the RBI's own directions keep it there.
2. **What does exist is a dense set of official aggregates**, published monthly, quarterly or annually, at national, state, district and bank level: volumes and values per payment instrument, cards and terminals per bank, merchant-category and state shares for UPI, and, since September 2022, a **monthly fraud series with a fraud rate per transaction**. That is enough to calibrate a generator and to set the prevalence, ticket sizes, instrument mix, geography and fraud typology it should reproduce.
3. **Real Indian fraud is far rarer per transaction than the European dataset Prism was built on.** RBI's registry reports one fraudulent payment per **80,847** transactions in July 2026 (0.0012%), against one in 578 (0.173%) in the ULB card data. Whatever we build must be evaluated at Indian prevalence, which changes thresholds, class weights and how results are reported.
4. **The fraud that matters in India by value is not card-present fraud.** By the government's own numbers, investment scams took 76% of money lost in 2025, and 98.5% of reported fraud value sits in transactions above Rs 10,000. By count, UPI fraud dominates complaints (47% of all cyber crime complaints 2020 to mid-2023) and card fraud is a distant second (11%). A generator that only does card testing is modelling the wrong India.

## 2. The sources, one by one

### 2.1 Reserve Bank of India

**Payment System Indicators (PSI), monthly.** `https://www.rbi.org.in/Scripts/PSIUserView.aspx` (July 2026 is `?Id=62`). Five parts: payment and settlement statistics, payment modes and channels, payment infrastructures, international modes, and **Domestic Payment Frauds**. Excel and PDF per month; the Excel is the file to script against.

July 2026 volumes and values (from the July 2026 page):

| Instrument | Volume (million) | Value (Rs crore) | Average ticket (Rs) |
|---|---|---|---|
| UPI | 23,658 | 29,87,880 | 1,263 |
| Credit cards | 605 | 2,08,186 | 3,440 |
| Debit cards | 102 | 37,932 | 3,735 |
| IMPS | 364 | 7,11,704 | 19,534 |
| PPIs (wallets, prepaid cards) | 925 | 28,607 | 309 |

**Domestic Payment Frauds table (PSI part 5).** Monthly since September 2022: fraud volume (lakh), fraud value (Rs crore), fraud frequency ("one in every N transactions"), and FTS (fraud value per 10,000 rupees of payment value, in basis points). Footnote: "Data on domestic payment fraud statistics as reported by Scheduled Commercial Banks (excluding RRBs), Non-bank PPI issuers and Non-bank Credit Card issuers in CPFIR (Central Payments Fraud Information Registry)" and "Data does not include attempts to perpetrate frauds." Full series as read on 23 September 2026:

| Month | Frauds (lakh) | Value (Rs crore) | One fraud per N transactions | FTS (bps) |
|---|---|---|---|---|
| Sep 2022 | 1.71 | 249 | 58,177 | 0.127 |
| Oct 2022 | 1.79 | 220 | 59,533 | 0.127 |
| Nov 2022 | 2.06 | 257 | 50,621 | 0.143 |
| Dec 2022 | 1.54 | 204 | 72,427 | 0.103 |
| Jan 2023 | 1.57 | 195 | 71,681 | 0.106 |
| Feb 2023 | 2.29 | 317 | 47,018 | 0.177 |
| Mar 2023 | 2.25 | 333 | 53,907 | 0.142 |
| Apr 2023 | 1.75 | 273 | 68,927 | 0.152 |
| May 2023 | 2.03 | 285 | 63,052 | 0.147 |
| Jun 2023 | 1.74 | 265 | 72,555 | 0.128 |
| Jul 2023 | 2.24 | 286 | 59,899 | 0.146 |
| Aug 2023 | 2.40 | 320 | 58,392 | 0.157 |
| Sep 2023 | 2.52 | 366 | 55,058 | 0.174 |
| Oct 2023 | 2.23 | 335 | 66,950 | 0.164 |
| Nov 2023 | 2.57 | 428 | 57,619 | 0.209 |
| Dec 2023 | 2.92 | 432 | 53,327 | 0.187 |
| Jan 2024 | 2.69 | 435 | 56,336 | 0.207 |
| Feb 2024 | 2.57 | 507 | 61,163 | 0.231 |
| Mar 2024 | 2.57 | 471 | 67,394 | 0.177 |
| Apr 2024 | 2.38 | 414 | 70,682 | 0.191 |
| May 2024 | 2.81 | 545 | 62,637 | 0.240 |
| Jun 2024 | 2.36 | 480 | 73,787 | 0.206 |
| Jul 2024 | 2.54 | 462 | 71,157 | 0.194 |
| Aug 2024 | 2.34 | 401 | 79,284 | 0.170 |
| Sep 2024 | 2.19 | 401 | 84,751 | 0.158 |
| Oct 2024 | 2.35 | 411 | 87,720 | 0.159 |
| Nov 2024 | 2.03 | 345 | 93,299 | 0.155 |
| Dec 2024 | 2.59 | 425 | 78,852 | 0.155 |
| Jan 2025 | 2.30 | 361 | 90,183 | 0.140 |
| Feb 2025 | 1.83 | 288 | 107,459 | 0.121 |
| Mar 2025 | 1.84 | 291 | 121,303 | 0.094 |
| Apr 2025 | 1.80 | 307 | 119,660 | 0.121 |
| May 2025 | 1.87 | 329 | 120,406 | 0.129 |
| Jun 2025 | 1.87 | 303 | 118,068 | 0.111 |
| Jul 2025 | 2.87 | 440 | 81,445 | 0.160 |
| Aug 2025 | 2.40 | 411 | 100,736 | 0.166 |
| Sep 2025 | 3.03 | 431 | 77,799 | 0.150 |
| Oct 2025 | 2.81 | 407 | 88,210 | 0.146 |
| Nov 2025 | 2.42 | 356 | 101,244 | 0.139 |
| Dec 2025 | 2.54 | 408 | 101,153 | 0.130 |
| Jan 2026 | 2.57 | 383 | 100,143 | 0.134 |
| Feb 2026 | 2.61 | 383 | 92,435 | 0.137 |
| Mar 2026 | 2.59 | 337 | 103,466 | 0.097 |
| Apr 2026 | 2.87 | 410 | 91,287 | 0.141 |
| May 2026 | 2.61 | 385 | 103,702 | 0.137 |
| Jun 2026 | 3.53 | 489 | 75,720 | 0.150 |
| Jul 2026 | 3.43 | 458 | 80,847 | 0.142 |

What the series says: 1.5 to 3.5 lakh frauds a month on 20 to 30 billion payments; a fraud rate that fell from roughly 1 in 50,000 (early 2023) to 1 in 100,000 to 120,000 (2025) and drifted back to 1 in 75,000 to 80,000 in mid-2026; an average fraud of about Rs 13,000 (Rs 458 crore over 3.43 lakh cases in July 2026); and fraud value at about 0.0014% of payment value.

**Bankwise ATM/POS/Card Statistics, monthly.** `https://www.rbi.org.in/scripts/atmview.aspx` (July 2026 is `atmid=185`; archives to January 2000). Per bank (about 60 banks across public, private, foreign, payments and small finance): ATMs and CRMs on and off site, POS terminals, micro ATMs, Bharat QR and UPI QR codes, cards outstanding, and credit and debit card transaction volume and value at POS, online and ATM. Excel per month. July 2026 headline figures as reported from this series: 12.29 crore credit cards outstanding (+9.9% year on year), credit card spend Rs 2.08 lakh crore on 60.5 crore transactions, 1.03 billion debit cards.

**Annual Report 2025-26 (published May 2026), fraud chapter.** Frauds reported by banks and financial institutions under the Fraud Monitoring Return: 10,114 cases, Rs 48,021 crore in 2025-26 (against 23,722 cases, Rs 32,803 crore in 2024-25), of which 314 legacy cases worth Rs 30,199 crore were re-reported after the Supreme Court's March 2023 judgement. Card, internet and digital payments category: 293 cases, Rs 29 crore in 2025-26; 13,332 cases, Rs 517 crore in 2024-25; 28,836 cases, Rs 1,452 crore in 2023-24. By bank group in 2025-26: public sector 5,418 cases, Rs 35,709 crore; private 3,956, Rs 11,399 crore; foreign 210, Rs 290 crore; small finance 467, Rs 114 crore; payments banks 47, Rs 11 crore.

The collapse from 13,332 to 293 digital cases is a reporting change, not a fraud change: the **Master Directions on Fraud Risk Management (15 July 2024)** require notice, an opportunity to represent and a reasoned order before an account is classified as fraud, and set the reporting threshold to law enforcement at Rs 1 lakh. The Lok Sabha answer of 3 August 2026 (Unstarred Q 2432) states it directly: "Post issuance of revised RBI Master Directions on Fraud Risk Management dated July 15, 2024, the banks are reporting only those payment system related transactions which are concluded as fraud." The CPFIR series above is on a different, all-amounts, per-transaction basis and shows no such drop.

**Lok Sabha Unstarred Q 2432, 3 August 2026: digital payment frauds by instrument, FY 2021-22 to FY 2025-26** (RBI data as tabled):

| Category | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|---|
| Credit cards | 21,610 / Rs 86.06 cr | 36,477 / Rs 189.45 cr | 118,106 / Rs 812.89 cr | 66,106 / Rs 336.07 cr | 1,692 / Rs 20.16 cr |
| Debit or ATM cards | 26,876 / Rs 88.54 cr | 11,711 / Rs 54.28 cr | 37,106 / Rs 189.06 cr | 18,882 / Rs 67.86 cr | 521 / Rs 1.65 cr |
| Internet banking | 17,404 / Rs 77.47 cr | 27,582 / Rs 176.40 cr | 138,027 / Rs 1,058.80 cr | 59,337 / Rs 412.52 cr | 7 / Rs 0.19 cr |
| UPI | | | 141 / Rs 0.64 cr | 355 / Rs 1.73 cr | |
| Wallets, prepaid | | | 5 / Rs 0.01 cr | 9 / Rs 0.02 cr | |
| IMPS, NEFT, RTGS | | | 12 / Rs 3.37 cr | 74 / Rs 5.55 cr | |
| AEPS, Aadhaar | | | 366 / Rs 0.32 cr | 3,319 / Rs 2.19 cr | |
| Other | | | 6 / Rs 1.09 cr | 20 / Rs 4.37 cr | |
| Total | | | 293,239 / Rs 2,060.75 cr | 144,855 / Rs 821.88 cr | 5,997 / Rs 35.86 cr |

Read with care: this is the FMR series (bank-reported, post-July-2024 concluded frauds only), so the UPI rows are tiny while the Ministry of Finance's separate UPI series (below) counts lakhs of incidents. The 2023-24 row is the last clean year for instrument mix: internet banking 47%, credit cards 40%, debit and ATM cards 13% of cases.

**Payment Systems Report, half-yearly** (December 2024, June 2025, December 2025 editions; `rbidocs.rbi.org.in`). Volume and value by system over five years, infrastructure, and a fraud section built on CPFIR. Useful for the narrative; the PSI table above carries the same fraud numbers monthly.

**Discussion Paper "Exploring safeguards in digital payments to curb frauds", 9 April 2026.** Cites NCRP (the I4C portal) fraud reports: 2021, 2.6 lakh reports, Rs 551 crore; 2022, 6.9 lakh, Rs 2,290 crore; 2023, 13.1 lakh, Rs 7,465 crore; 2024, 24 lakh, Rs 22,848 crore; 2025, 28 lakh, Rs 22,931 crore. Transactions above Rs 10,000 are about 45% of reported fraud cases by volume but 98.5% by value; about 92% of fraud value is above Rs 50,000. Proposals: a one-hour lag on authorised push payments above Rs 10,000 (merchant payments and recurring mandates excluded), trusted-person authentication above Rs 50,000 for citizens aged 70 and over, a Rs 25 lakh annual ceiling on credits into low-turnover accounts, and a customer kill switch across UPI, cards, net banking and wallets.

**Ombudsman Scheme Annual Report 2024-25 (December 2025).** 13.34 lakh complaints, up 13.55%. By category: loans and advances 86,670; credit cards 50,811; mobile and electronic banking 49,951; deposit accounts 49,913; ATM and debit cards 18,082; others 30,760 (category counts as mirrored from the report's table). Credit-card complaints rose 20%; mobile and electronic banking complaints fell 12.74%.

**BE(A)WARE booklet (7 March 2022).** Fourteen modus operandi for banks, compiled from Ombudsman complaints: phishing links, vishing calls, online sales-platform fraud (the UPI "request money" trick), unverified mobile apps, ATM card skimming, screen sharing and remote access, SIM swap and cloning, credential compromise via search engines, QR code scams, social media impersonation, juice jacking, lottery fraud, online job fraud, money mules; plus NBFC variants (fake loan apps and websites, OTP-based loan fraud, Ponzi schemes).

**Rules that shape the traffic** (all RBI or NPCI): mandatory additional factor of authentication on domestic card-not-present payments; contactless card payments PIN-free up to Rs 5,000; card-on-file and device tokenisation (2019, 2021) so merchants hold no PANs; customer-controlled card switches and limits (2020); UPI P2P Rs 1 lakh per day; UPI person-to-merchant up to Rs 5 lakh per transaction and Rs 10 lakh per day for verified merchants in capital markets, insurance, travel, loan repayment, tax, hospital and education categories since 15 September 2025; the new risk-based authentication directions (2025).

### 2.2 NPCI

**UPI product statistics, monthly** (`npci.org.in/product/upi/product-statistics`) and **UPI ecosystem statistics** (`npci.org.in/what-we-do/upi/upi-ecosystem-statistics`, with archives): monthly volume and value, banks live, app-wise (PhonePe 46.2%, Google Pay 32.7% of May 2026 volume), PSP-wise, remitter and beneficiary bank-wise, and since 2025 state-wise product statistics and a 29-category merchant classification. The pages are rendered in the browser; the monthly files are downloadable there but not by script, so the figures below are taken from SBI Research (August 2025) and Worldline (November 2025), both of which cite NPCI. Also RuPay, IMPS, AePS, NETC and BBPS statistics on the same site.

- Volume and value: 185.9 billion UPI transactions worth Rs 260.6 lakh crore in FY 2024-25 (+41.7% and +30.3%); 23.66 billion in July 2026 alone (PSI above); 675 million a day in August 2025, average daily value Rs 90,446 crore (SBI Research from NPCI).
- P2M versus P2P (SBI Research, July 2025): merchant payments are about 64% of UPI volume and 29% of value; the rest is person to person. Worldline (1H 2025): 67.01 billion P2M transactions, average P2M ticket Rs 592; overall average ticket Rs 1,348, down from Rs 1,478 a year earlier.
- Merchant categories (NPCI's 29 published MCC groups, July 2025, via SBI Research): they cover 82% of volume and 52% of value; the top 15 are 70% of volume and 47% of value; groceries alone are 24.3% of transactions and 8.8% of value; "debt collection agencies" (loan EMIs collected by NBFCs and fintechs) are 12.8% of value on 1.3% of volume with a Rs 5,817 ticket. Worldline's acquiring data: grocery, restaurants, clothing, hospitals, hotels, pharmacies, jewellery and department stores are 62% of volume and 58% of value.
- Geography (NPCI state-wise, July 2025): Maharashtra 9.8% of volume, Karnataka 5.5%, Uttar Pradesh 5.3%.
- Users: about 500 million UPI users and some 65 million merchants (industry estimates; NPCI publishes banks live, 703 in FY26, not users).

**UPI fraud series (Ministry of Finance to Lok Sabha, tabled 25 November 2024 and 15 December 2025):** FY 2023-24, 13.42 lakh incidents, Rs 1,087 crore; FY 2024-25, 12.64 lakh, Rs 981 crore (an earlier answer gave 6.32 lakh and Rs 485 crore to September 2024); FY 2025-26 April to November, 10.64 lakh, Rs 805 crore. Average about Rs 7,500 to 8,000 per incident; against 185.9 billion FY25 transactions that is roughly 6.8 incidents per million UPI transactions.

### 2.3 Government of India: I4C, NCRB, DoT, MeitY

**I4C, National Cyber Crime Reporting Portal (NCRP) and CFCFRMS (helpline 1930).** Parliamentary answers, Ministry of Home Affairs: Lok Sabha Unstarred Q 452 (2 December 2025) and Rajya Sabha Unstarred Q 1349 (11 February 2026), both at `mha.gov.in/MHA1/Par2017/pdfs/`. Complaints: 4.52 lakh (2021), 19.18 lakh (2024), 28.15 lakh (2025). CFCFRMS has saved more than Rs 8,189 crore in more than 23.61 lakh complaints as on 31 December 2025 (Rs 7,130 crore and 23.02 lakh at 2 December 2025); 11.14 lakh SIM cards and 2.96 lakh IMEIs blocked; the Suspect Registry launched 10 September 2024 holds 18.43 lakh suspect identifiers from banks and 24.67 lakh layer-1 mule accounts, and Rs 8,031.56 crore of transactions have been rejected against it. Category intelligence tabled in 2025: investment scams were 76% of money lost on 35% of complaints; digital arrest frauds rose from 39,925 cases (Rs 91 crore) in 2022 to 123,672 (Rs 1,935 crore) in 2024. Actual restitution to victims lags freezes (about Rs 167 crore paid back against Rs 8,690 crore frozen by January 2026, per CyberPeace's reading of the same answers).

**NCRB, Crime in India (annual; 2023 edition released October 2025; `ncrb.gov.in`).** Cyber crimes registered: 44,735 (2019), 50,035 (2020), 52,974 (2021), 65,893 (2022), 86,420 (2023). Cases registered under "Fraud" (IPC, cyber): 6,229, 10,395, 14,007, 17,470, 19,466 for 2019 to 2023; a separate motive-based count puts fraud at 68.9% of all 2023 cyber crime cases (59,526). Fraud sub-heads (Rajya Sabha Q 1349, from NCRB):

| Sub-head | 2021 | 2022 | 2023 |
|---|---|---|---|
| Credit or debit card | 1,624 | 1,665 | 2,098 |
| ATMs | 1,899 | 1,690 | 1,783 |
| Online banking fraud | 4,823 | 6,491 | 4,435 |
| OTP frauds | 2,028 | 2,910 | 5,116 |
| Others | 3,633 | 4,714 | 6,034 |
| Fraud total | 14,007 | 17,470 | 19,466 |

State-wise cyber crime cases 2019 to 2023 are in Annexure II of Lok Sabha Q 452 (Karnataka 21,889, Telangana 18,236, Uttar Pradesh 10,794 in 2023). NCRB figures are FIRs, two years behind, and a small fraction of NCRP complaints (86,420 cases against 15 to 19 lakh complaints in the same year), which measures how little of reported fraud becomes a criminal case.

**Department of Telecommunications: Digital Intelligence Platform (DIP) and Financial Fraud Risk Indicator (FRI), launched 22 May 2025.** A per-mobile-number risk grade (medium, high, very high) built from NCRP reports, the Chakshu platform and bank intelligence; more than 1,000 banks, TPAPs and payment operators onboarded; RBI advised banks to integrate it (PIB, June 2025); Rs 5,043.73 crore of customer money protected by August 2026 (Ministry of Communications).

**RBI Innovation Hub, MuleHunter.AI (December 2024).** Built on 19 behavioural signatures of mule accounts; 23 to 26 banks live by late 2025; no public statistics on accounts flagged (RBI declined an RTI request, per MediaNama, December 2025).

**Digital Payments Intelligence Platform (DPIP), operated by IDPIC with RBIH.** Smart Registry live on 1 September 2026 with eight banks; used for onboarding, profile changes, beneficiary additions and scrubbing of existing customers; Phase 2, real-time risk scores per transaction with explainability reports, in user acceptance testing; 25 more banks within two months and the top 15 payment aggregators by January 2027, covering about 80% of volume (MediaNama, September 2026).

**data.gov.in resources** (the Open Government Data platform; pages are viewable in a browser, the CSVs behind them refuse scripted download): "Year-wise Details of Incidents of Frauds and Total Amount Involved as reported on CPFIR from 2022-23 to 2024-25" (Rajya Sabha Q 1991, 18 March 2025); "Year-wise Details of Frauds ... under the Category of Card/Internet and Digital Payments from 2020-21 to 2024-25"; "State/UT-wise Number of Cases Registered under Fraud for Cyber Crimes such as Credit/Debit Cards, ATMs, Online Banking Frauds, OTP Frauds and Others from 2018-2022"; "Crime Head-wise and State/UT-wise Cyber Crimes, IT Act Cases during 2023".

### 2.4 Industry and open data

**PhonePe Pulse** (`github.com/PhonePe/pulse`, licence CDLA-Permissive-2.0). Quarterly from Q1 2018 to Q2 2026, national, state and district level, for transactions (categories P2P, Retail or Business, Utility), users (by device brand) and insurance; "aggregated", "map" and "top" variants. Current files carry counts only (Q2 2026 national: P2P 11.89 billion, Retail 24.69 billion, Utility 2.08 billion on PhonePe alone). This is the only public source with district-level transaction counts, and the only open-licensed one.

**Worldline India Digital Payments Report** (half-yearly; 1H 2025 published November 2025; PDF at `worldline.com/en-in`). Acquirer-side view: UPI 106.36 billion transactions worth Rs 143.34 lakh crore in 1H 2025 (+35%); average tickets as above; credit cards outstanding +23% between January 2024 and June 2025; monthly credit card spend above Rs 2.2 lakh crore with ticket size down 6%; merchant category and city rankings; "QR-first, PoS-as-needed" acceptance.

**SBI Research, "New Insights from UPI Data" (18 August 2025).** The most complete public reading of NPCI's state and MCC releases (figures used above); notes that NPCI's 29 categories leave 48% of value unclassified and that the ISO 18245 MCC list has about 300 codes.

**LocalCircles UPI fraud survey (March to June 2025; 32,000+ responses, 365 districts).** One in five families experienced a UPI fraud in three years; 51% of victims filed no complaint anywhere; of those hit, 50% reported someone taking control of UPI settings or PIN, 40% clicked a fraudulent payment link, 20% a malicious QR code, 20% gave an OTP or PIN to a caller posing as bank staff. Self-reported, so use it for the shape of under-reporting, not for rates.

**Future Crime Research Foundation (IIT Kanpur incubated), "A Deep Dive into Cybercrime Trends Impacting India" (September 2023; NCRP data January 2020 to June 2023).** Online financial fraud 77.41% of complaints; UPI fraud 47.25% of all complaints; debit, credit card and SIM-swap fraud 11.27%.

### 2.5 What is not public, and why it matters

- Transaction-level data with labels, from any Indian institution. CPFIR is a registry, not a dataset; NPCI switch logs are not released; banks are bound by RBI's outsourcing and data-protection directions and the DPDP Act 2023.
- Hour-of-day or day-of-week distributions of payments. No official series; only trade commentary.
- Per-customer sequences (the velocity and history features every real detector uses). PhonePe Pulse is the closest, and it is district-quarter counts.
- The fraud typology mix inside CPFIR (card-present versus card-not-present versus account takeover). Only the FMR instrument table and the complaint-based shares exist.
- Any card-present skimming rate. NCRB's ATM and card sub-heads are the only counts.

The three official fraud series measure different things and must never be mixed: the RBI Annual Report FMR series (bank-classified frauds, mostly above Rs 1 lakh, concluded after due process since July 2024); the RBI PSI or CPFIR series (every reported fraudulent payment transaction, all amounts, monthly, per-transaction rate available); and the I4C NCRP series (citizen complaints of all cyber financial fraud including investment and impersonation scams, most of which are not payment-system frauds at all).

## 3. Calibration targets for the generator

These are the numbers the Indian stream should reproduce, each with the source it comes from. A generator that can be checked against them is grounded; one that cannot is decoration.

| What | Target | Source |
|---|---|---|
| Fraud prevalence per transaction | 1 in 80,000 to 1 in 120,000 (0.0008% to 0.0012%), month-varying as in the PSI table; demo modes may boost, but the label says so | RBI PSI Domestic Payment Frauds, 2025-26 |
| Fraud value as share of payment value | 0.10 to 0.16 bps | same |
| Average fraud amount | about Rs 13,000 overall; about Rs 7,500 for UPI incidents; 45% of cases above Rs 10,000 carrying 98.5% of value | PSI July 2026; MoF UPI series; RBI discussion paper |
| Instrument mix of legitimate traffic (by count) | UPI about 85% of retail volume; credit cards 2.3%; debit cards 0.4%; PPIs 3.5%; IMPS 1.4% (July 2026) | RBI PSI |
| Ticket sizes | UPI overall Rs 1,263 to 1,348, P2M Rs 592; credit card Rs 3,440; debit card Rs 3,735; PPI Rs 309; IMPS Rs 19,500 | RBI PSI July 2026; Worldline 1H 2025 |
| UPI P2M versus P2P | 64% of volume, 29% of value are merchant payments | SBI Research from NPCI, July 2025 |
| Merchant categories | groceries 24% of UPI volume, 9% of value; top 15 categories 70% of volume, 47% of value; debt collection 13% of value at Rs 5,817 | NPCI via SBI Research |
| Geography | Maharashtra 9.8%, Karnataka 5.5%, UP 5.3% of UPI volume; district counts from PhonePe Pulse | NPCI; PhonePe Pulse |
| Cards and acceptance | 12.29 crore credit cards, 1.03 billion debit cards; card and terminal counts per bank monthly | RBI card statistics |
| Fraud typology mix (by count of complaints) | UPI-rail social engineering first, card and SIM-swap second (11%), OTP frauds the fastest-growing NCRB sub-head (2,028 to 5,116 in two years) | FCRF; NCRB |
| Fraud typology mix (by value) | investment and impersonation scams dominate (76% of 2025 losses); these are authorised push payments, not card fraud | I4C via MHA |
| Behaviours to model | the 14 BE(A)WARE modus operandi; mule-account layering (24.67 lakh layer-1 mule accounts on the Suspect Registry); a per-number risk grade (FRI) as a feature banks now receive | RBI; I4C; DoT |
| Rules that constrain legitimate traffic | AFA on domestic CNP; Rs 5,000 contactless limit; Rs 1 lakh per day UPI P2P; category-specific P2M limits; tokenised card-on-file | RBI; NPCI |
| Reporting realism | attempts are excluded from CPFIR; frauds are reported with delay; most victims (51%) never complain | RBI PSI footnote; LocalCircles |

Two consequences for the generator in particular. First, its current default prevalence (3% in the CLI exports we tested) is about 2,500 times the Indian rate; a demo mode may keep it, but the calibrated mode has to sit at the PSI rate. Second, its playbooks are card-centric (probes, ATO, bust-out, enumeration); the Indian complaint data says the mass of fraud is authorised push payment on UPI, where the victim authorises the transaction. That is a different signal problem (the transaction is legitimate from the rail's point of view; the tells are counterparty, velocity, new-beneficiary and mule-account behaviour), and it is exactly what the RBI's April 2026 proposals, the FRI and DPIP are aimed at.

## 4. What this means for the fraud-detection track

1. **Evaluate at Indian prevalence.** With one fraud in 80,000, the no-skill PR-AUC baseline is 0.0000124, thresholds tuned at 0.17% are wrong by two orders of magnitude, and precision at any operating point collapses unless the model is very sharp. Alerts per caught fraud and the investigator budget become the honest metrics; the RBI's own "one in N transactions" is the number to report beside PR-AUC.
2. **Report cost, not just rank.** The discussion paper's thresholds (Rs 10,000 and Rs 50,000) and value concentration (98.5% above Rs 10,000) mean a detector should be judged on value caught, not count caught; a model that misses the 55% of small frauds but catches the large ones saves most of the money.
3. **Named features, real semantics.** The Indian stream has instrument, channel, merchant category, amount band, counterparty age, new-beneficiary, velocity, cross-border, AFA outcome and a number-risk grade. Prism's explanation layer, which today speaks only in measured facts because V1 to V28 have no meaning, can say "first payment to this beneficiary, 3 minutes after an OTP request, amount above your 30-day maximum". That is the explainability the project promised.
4. **Two families of fraud, two detectors.** Card-rail fraud (unauthorised: skimming, CNP, ATO) fits the current supervised and anomaly models. Authorised push payment fraud needs counterparty and sequence features and a different label (the payer is the victim, the transaction is authorised); a graph or velocity component over payer, payee and device is the natural addition, and it is what MuleHunter and DPIP score.
5. **Time-ordered evaluation, with drift.** The PSI series shows the fraud rate halving over 2024 and rising again in 2026; modus operandi change faster (digital arrest went from 40,000 to 124,000 cases in two years). Random splits hide this. Train on months, test on later months, and report the decay.
6. **Keep ULB.** It stays the only real transaction-level fraud data we have and the evidence that the pipeline works on real data. The Indian stream demonstrates the deployed form; the report says which is which.

## 5. How the three deliverables fit together for the teacher

- **Research (this document):** what India publishes, what it does not, what the numbers are, and why no public Indian transaction dataset exists. The answer to "why not just use real Indian data" is section 2.5.
- **Generator:** a simulator whose legitimate traffic is calibrated to RBI, NPCI and PhonePe Pulse aggregates and whose fraud follows RBI's own typology list at RBI's own rate, with a leak gate so no column gives the label away, and a run record stating every target it was checked against.
- **Detection (Prism):** the same three models and explanations, evaluated at Indian prevalence and in time order, on real ULB data and on the calibrated Indian stream, with the customer-facing explanation in plain English on named Indian fields.

## 6. Access notes

| Source | Access | Scriptable | Licence or terms |
|---|---|---|---|
| RBI Payment System Indicators | rbi.org.in, monthly Excel and PDF | page HTML yes; rbidocs PDFs sit behind a bot check | RBI site terms; public statistics |
| RBI bankwise card statistics | rbi.org.in `atmview.aspx`, monthly Excel | as above | public statistics |
| RBI Annual Report, Ombudsman report, discussion papers | rbidocs PDFs | bot check; download in a browser | public |
| RBI DBIE | data.rbi.org.in, time series | portal, RBIDATA app | public |
| NPCI statistics | npci.org.in pages | browser only (403 to scripts) | public |
| MHA parliamentary answers | mha.gov.in PDFs | yes with a browser user agent | public |
| PIB releases | pib.gov.in | yes | public |
| NCRB Crime in India | ncrb.gov.in PDFs (Part I is 22 MB; cyber tables are in the later volumes) | yes | public |
| data.gov.in | resource pages; CSV downloads | page yes; files refuse scripted download | Government Open Data Licence India |
| PhonePe Pulse | GitHub, JSON | yes (`gh api` or raw) | CDLA-Permissive-2.0 |
| Worldline, SBI Research, FCRF, LocalCircles | PDFs and pages | yes | publisher terms; cite |

## Sources

- RBI Payment System Indicators, July 2026: https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=62 and index https://www.rbi.org.in/Scripts/PSIUserView.aspx
- RBI Bankwise ATM/POS/Card Statistics: https://www.rbi.org.in/scripts/atmview.aspx (July 2026: https://www.rbi.org.in/Scripts/ATMView.aspx?atmid=185)
- RBI Payment Systems Report, December 2024: https://rbidocs.rbi.org.in/rdocs/Publications/PDFs/PSR270120251BEE95CF47F2426B9740075D405FA070.PDF
- RBI Discussion Paper, Exploring safeguards in digital payments to curb frauds, 9 April 2026 (mirror): https://www.medianama.com/wp-content/uploads/2026/04/DP090420261ED5D6E68D724A6EA870B7E68E45F80F.pdf
- RBI Master Directions on Fraud Risk Management, 15 July 2024, FAQs: https://www.rbi.org.in/commonman/english/scripts/FAQs.aspx?Id=3763
- RBI Annual Report 2025-26 fraud figures (PTI via ETV Bharat, 29 May 2026): https://www.etvbharat.com/en/bharat/financial-institutions-report-over-several-cases-of-fraud-in-fy26-says-rbi-data-enn26052903021
- RBI card statistics, July 2026 headline (ICICI Direct): https://www.icicidirect.com/research/equity/trending-news/india-credit-cards-hit-1229-crore-in-july-as-spends-rise-74-to-208-lakh-crore
- Lok Sabha Unstarred Q 2432, 3 August 2026, digital payment frauds by category (TaxGuru transcription): https://taxguru.in/rbi/government-shares-digital-payment-fraud-data-rbi-anti-fraud-measures-lok-sabha.html
- RBI Ombudsman Scheme Annual Report 2024-25 categories (Dataful mirror of the report table): https://dataful.in/datasets/153/ and summary https://affairscloud.com/rbis-annual-report-of-ombudsman-scheme-2024-25-complaints-up-13-55-loans-and-credit-cards-lead-grievances/
- RBI BE(A)WARE booklet press release, 7 March 2022: https://rbidocs.rbi.org.in/rdocs/PressRelease/PDFs/PR1817F4A7B3662BEC4AD7AC6A2D85EAF937ED.PDF and https://taxguru.in/rbi/rbi-beaware-aware-beware-booklet-frauds.html
- NPCI UPI product statistics: https://www.npci.org.in/product/upi/product-statistics and ecosystem statistics https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics
- NPCI P2M limits, 15 September 2025: https://affairscloud.com/npci-raised-per-upi-transaction-limit-to-rs-5-lakh-and-cumulative-limit-for-24-hrs-to-rs-10-lakh/
- SBI Research, New Insights from UPI Data, 18 August 2025: https://sbi.bank.in/documents/13958/14472/New+Insights+from+UPI+Data_SBI+Research.pdf/5e8227bf-78b1-5838-59b5-1ba3e76099b5?t=1755680722524
- Worldline India Digital Payments Report 1H 2025: https://worldline.com/content/dam/worldline/local/en-in/documents/main-page11111/Worldline-India-Digital-Payment-Report-1H-2025-1.pdf and press release https://worldline.com/en-in/home/top-navigation/media-relations/press-releases/worldline-releases-india-digital-payments-report-for-1H-2025
- Ministry of Finance UPI fraud series (Lok Sabha, 25 November 2024): https://www.medianama.com/2024/11/223-upi-fraud-up-85-per-cent-fy2023-24-finance-ministry-data/ and (15 December 2025): https://the420.in/india-upi-fraud-data-fy26-parliament-digital-payments/
- MHA, Lok Sabha Unstarred Q 452, 2 December 2025: https://www.mha.gov.in/MHA1/Par2017/pdfs/par2025-pdfs/LS02122025/452.pdf
- MHA, Rajya Sabha Unstarred Q 1349, 11 February 2026: https://www.mha.gov.in/MHA1/Par2017/pdfs/par2026-pdfs/RS11022026/1349.pdf
- CyberPeace analysis of Rajya Sabha proceedings, 30 July 2025: https://cyberpeace.org/resources/blogs/cyberpeace-analysis-indias-cybercrime-surge-signals-a-growing-digital-security-challenge-an-assessment-based-on-rajya-sabha-proceedings-and-mha-data
- PIB, cyber fraud and NCRB 2022 categories (PRID 2080186): https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2080186
- NCRB Crime in India 2023: https://www.ncrb.gov.in/uploads/files/1CrimeinIndia2023PartI.pdf
- data.gov.in, CPFIR incidents 2022-23 to 2024-25: https://www.data.gov.in/resource/year-wise-details-incidents-frauds-and-total-amount-involved-reported-central-payments
- data.gov.in, state-wise cyber fraud sub-categories 2018-2022: https://www.data.gov.in/resource/stateut-wise-number-cases-registered-under-fraud-cyber-crimes-such-creditdebit-cards-atms
- data.gov.in, cyber crimes IT Act cases 2023: https://www.data.gov.in/resource/crime-head-wise-and-stateut-wise-cyber-crimes-it-act-cases-during-2023
- DoT Financial Fraud Risk Indicator: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2130249&reg=48&lang=2 ; RBI advisory: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2141616&reg=3&lang=2 ; DIP onboarding: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2207376&reg=3&lang=1 ; Rs 5,043.73 crore protected: https://upstox.com/news/business-news/latest-updates/govt-s-fraud-risk-tool-prevents-suspected-cyber-fraud-transactions-worth-over-5-000-crore/article-200002/
- RBIH MuleHunter.AI: https://rbihub.in/projects/mulehunter and https://www.medianama.com/2025/12/223-rti-23-banks-mulehunter-mule-accounts/
- DPIP Smart Registry, 1 September 2026: https://www.medianama.com/2026/09/223-dpip-smart-registry-fraud-prevention/
- PhonePe Pulse: https://github.com/PhonePe/pulse
- LocalCircles UPI fraud survey, 2025: https://www.localcircles.com/a/press/page/upi-fraud-complaint
- FCRF, cybercrime trends 2020 to 2023: https://futurecrime.org/ and https://www.theregister.com/2023/09/21/india_cybercrime_trends_report/
- CPFIR coverage extension, March 2025 (RBI Annual Report 2024-25 via MediaNama): https://www.medianama.com/2025/06/223-rbi-annual-report-2024-25-central-bank-agenda/
