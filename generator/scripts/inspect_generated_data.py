"""Inspection script for generated synthetic datasets."""

import csv
from collections import Counter
from pathlib import Path


def inspect_file(name: str, file_path: Path) -> None:
    print("=" * 80)
    print(f"DATASET INSPECTION REPORT: {name} ({file_path})")
    print("=" * 80)

    with open(file_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total = len(reader)
    frauds = [r for r in reader if r["is_fraud"] == "1"]
    legit = [r for r in reader if r["is_fraud"] == "0"]
    disputes = [r for r in reader if r.get("dispute_status", "") not in ("", "NONE")]

    print(f"Total Transactions: {total}")
    print(f"Legitimate Transactions: {len(legit)} ({len(legit)/total*100:.1f}%)")
    print(f"Fraud Transactions: {len(frauds)} ({len(frauds)/total*100:.1f}%)")
    print(f"Disputes Generated: {len(disputes)}")

    print("\n--- Product Distribution ---")
    product_counts = Counter(r["product_id"] for r in reader)
    for prod, count in product_counts.most_common():
        print(f"  {prod:<35}: {count:>4} ({count/total*100:>4.1f}%)")

    print("\n--- ISO 8583 Response Code Distribution ---")
    resp_counts = Counter(r["response_code"] for r in reader)
    for resp, count in resp_counts.most_common():
        print(f"  ISO {resp:<5}: {count:>4} ({count/total*100:>4.1f}%)")

    print("\n--- Channel Breakdown ---")
    channel_counts = Counter(r["channel_type"] for r in reader)
    for ch, count in channel_counts.most_common():
        print(f"  {ch:<25}: {count:>4} ({count/total*100:>4.1f}%)")

    print("\n--- Adversarial Scenarios Encountered ---")
    scen_counts = Counter(r["scenario_tag"] for r in frauds)
    for sc, count in scen_counts.most_common():
        print(f"  {sc:<35}: {count:>4} ({count/len(frauds)*100:>4.1f}%)")

    print("\n--- Dispute & Recovery Adjudication ---")
    disp_counts = Counter(r.get("dispute_status", "NONE") for r in frauds)
    for st, count in disp_counts.most_common():
        print(f"  {st:<35}: {count:>4}")

    if name == "US":
        ce3_count = sum(1 for r in frauds if r.get("ce3_qualified") == "True")
        arb_fee_total = sum(float(r.get("arbitration_fee_usd", 0.0)) for r in disputes)
        print(f"  Visa CE 3.0 Pre-Dispute Deflections: {ce3_count}")
        print(f"  Total Network Arbitration Fees Incurred: ${arb_fee_total:,.2f}")
    else:
        rbi_tiers = Counter(r.get("rbi_liability_tier", "") for r in disputes)
        print("  RBI Limited Customer Liability Tiers:")
        for tier, count in rbi_tiers.most_common():
            print(f"    {tier:<30}: {count:>3}")
        liens = Counter(r.get("cfcfrms_1930_lien_status", "") for r in disputes)
        print("  CFCFRMS 1930 Golden Hour Cyber Lien Outcomes:")
        for lien, count in liens.most_common():
            print(f"    {lien:<30}: {count:>3}")

    print("\n" + "-" * 80)
    print("SAMPLE DETAILED ADVERSARIAL DISPUTE RECORDS:")
    for d in disputes[:3]:
        print(f"  TX ID: {d['transaction_id']} | Card: {d['card_id']} ({d['product_id']})")
        print(f"    Amount: {d['amount']} {d['currency']} (Minor: {d['amount_minor']}) | MCC: {d['mcc']}")
        print(f"    Channel: {d['channel_type']} | ISO Resp: {d['response_code']} | ECI: {d.get('eci', '')} | 3DS: {d.get('trans_status_3ds', '')}")
        print(f"    Adversarial Attack: {d['scenario_tag']}")
        print(f"    Dispute Status: {d.get('dispute_status')} | Reason Code: {d.get('dispute_reason_code')}")
        if name == "US":
            print(f"    Visa CE 3.0 Qualified: {d.get('ce3_qualified')} | Arb Fee: ${float(d.get('arbitration_fee_usd', 0)):.2f}")
        else:
            print(f"    RBI Liability Tier: {d.get('rbi_liability_tier')} | 1930 Cyber Lien: {d.get('cfcfrms_1930_lien_status')}")
    print("\n")


if __name__ == "__main__":
    inspect_file("US", Path("data/sample_us.csv"))
    inspect_file("India", Path("data/sample_in.csv"))
