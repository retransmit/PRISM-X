"""Automated verification suite for Prism Generator Living Specifications.

Validates that:
1. All specification YAML files parse cleanly and adhere to schema definitions.
2. All probability distributions strictly sum to 1.0 (simplex constraints).
3. All financial units are non-negative and explicitly bounded (USD cents, INR paisa).
4. No ungrounded / sci-fi buzzwords (Rule 1 & Rule 4 Anti-Astronaut mandate) are present.
5. All adversarial playbooks maintain relational integrity with card products and human cohorts.
6. Post-authorization dispute lifecycles (Visa VCR, CE 3.0, arbitration fees) are grounded.
7. Merchant/Acquirer risk and scheme monitoring (VAMP, ECP, MATCH) adhere to network rules.
8. Cybercrime technical vectors (PEA additive guessing, VAAI, money mules) are empirically bounded.
9. Indian regulatory protections (RBI Customer Liability tiers, TAT compensation, PA Escrow, CFCFRMS 1930) are verified.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import yaml

SPEC_DIR = Path(__file__).resolve().parent.parent.parent / "spec"

FORBIDDEN_SCI_FI_BUZZWORDS = [
    "quantum",
    "loihi",
    "neuromorphic",
    "sheaf",
    "cohomology",
    "snark",
    "groth16",
    "plonk",
    "hyperbolic",
    "poincare",
    "riemannian",
    "monosemantic",
    "fastshap",
]


def load_yaml(filename: str) -> Dict[str, Any]:
    path = SPEC_DIR / filename
    assert path.exists(), f"Spec file {filename} does not exist in {SPEC_DIR}"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{filename} root must be a YAML mapping"
    return data


class TestTaxonomySpecifications:
    """Verifies physical, financial, and mathematical integrity of living specifications."""

    def test_spec_files_exist(self):
        """Verify all 5 core specification files exist."""
        required = [
            "01_financial_instruments.yaml",
            "02_human_personas.yaml",
            "03_payment_rail_gaps.yaml",
            "04_adversarial_playbooks.yaml",
            "05_india_payment_rails.yaml",
        ]
        for fname in required:
            assert (SPEC_DIR / fname).exists(), f"Missing canonical spec file: {fname}"

    def test_anti_astronaut_buzzword_veto(self):
        """Enforces Rule 1 & Rule 4: No speculative/sci-fi terms in specification files."""
        for spec_file in SPEC_DIR.glob("*.yaml"):
            content = spec_file.read_text(encoding="utf-8").lower()
            for buzzword in FORBIDDEN_SCI_FI_BUZZWORDS:
                assert buzzword not in content, (
                    f"VIOLATION: Forbidden buzzword '{buzzword}' detected in {spec_file.name}. "
                    "All specs must remain strictly grounded in empirical banking."
                )

    def test_indian_payment_rails_integrity(self):
        """Verifies statutory thresholds and product parameters for Indian rails."""
        data = load_yaml("05_india_payment_rails.yaml")
        assert "india_regulatory_rails" in data
        afa = data["india_regulatory_rails"]["additional_factor_of_authentication_afa"]

        # Statutory contactless limit ₹5,000.00 (500,000 Paisa)
        assert afa["exemptions"]["contactless_nfc_tap"]["pin_free_ceiling_paisa"] == 500000

        # Statutory e-Mandate standard limit ₹15,000.00
        assert afa["exemptions"]["e_mandate_recurring"]["standard_ceiling_paisa"] == 1500000

        # Offline micropayment ₹500.00
        assert afa["exemptions"]["offline_micropayments"]["per_tx_ceiling_paisa"] == 50000

        # Validate Indian Card Products
        products = data["indian_card_products"]
        assert len(products) >= 5, "Must define comprehensive Indian card categories"
        prod_ids = set()
        for prod in products:
            pid = prod["id"]
            assert pid not in prod_ids
            prod_ids.add(pid)
            if "credit_limit_paisa" in prod:
                lim = prod["credit_limit_paisa"]
                assert 0 < lim["min"] <= lim["median"] <= lim["max"]

        # Validate Indian Adversarial Playbooks
        playbooks = data["indian_adversarial_playbooks"]
        for pb in playbooks:
            for t_prod in pb["target_products"]:
                assert t_prod in prod_ids, f"Playbook {pb['id']} targets unknown product: {t_prod}"

    def test_financial_instruments_integrity(self):
        """Verifies credit limits, liability caps, and velocity boundaries on card products."""
        data = load_yaml("01_financial_instruments.yaml")
        assert "card_products" in data
        products = data["card_products"]
        assert len(products) >= 10, "Taxonomy must reflect comprehensive unconstrained spectrum"

        product_ids = set()
        for prod in products:
            pid = prod["id"]
            assert pid not in product_ids, f"Duplicate product ID: {pid}"
            product_ids.add(pid)

            # Validate limits
            if "credit_limit_cents" in prod:
                lim = prod["credit_limit_cents"]
                assert 0 < lim["min"] <= lim["median"] <= lim["max"], (
                    f"Inverted or non-positive credit limits for {pid}: {lim}"
                )

            if "daily_spend_limit_cents" in prod:
                assert prod["daily_spend_limit_cents"] > 0

            if "stip_floor_limit_cents" in prod:
                assert prod["stip_floor_limit_cents"] > 0

    def test_human_personas_probability_simplexes(self):
        """Verifies probability distributions in human persona models sum to 1.0."""
        data = load_yaml("02_human_personas.yaml")
        assert "consumer_clusters" in data
        clusters = data["consumer_clusters"]
        assert len(clusters) == 7, "Must contain all 7 empirical consumer cohorts from Fed DCPC"

        # Check population weights simplex
        total_pop_weight = sum(c["population_weight"] for c in clusters)
        assert math.isclose(total_pop_weight, 1.0, rel_tol=1e-4), (
            f"Population weights must sum to 1.0, got {total_pop_weight}"
        )

        for c in clusters:
            cid = c["id"]
            # Check channel mix simplex
            channels = c["primary_channels"]
            total_channel = sum(channels.values())
            assert math.isclose(total_channel, 1.0, rel_tol=1e-4), (
                f"Channel mix for {cid} must sum to 1.0, got {total_channel}"
            )

            # Check vigilance tier simplex
            vigilance = c["vigilance_tier_weights"]
            total_vigilance = sum(vigilance.values())
            assert math.isclose(total_vigilance, 1.0, rel_tol=1e-4), (
                f"Vigilance tier weights for {cid} must sum to 1.0, got {total_vigilance}"
            )

    def test_circadian_diurnal_simplexes(self):
        """Verifies von Mises mixture weights sum to 1.0 for weekday and weekend profiles."""
        data = load_yaml("02_human_personas.yaml")
        diurnal = data["circadian_diurnal_model"]

        # Weekday
        weekday_weights = [comp["weight"] for comp in diurnal["weekday_profile"]["components"]]
        assert math.isclose(sum(weekday_weights), 1.0, rel_tol=1e-4), (
            f"Weekday circadian mixture weights must sum to 1.0, got {sum(weekday_weights)}"
        )

        # Weekend
        weekend_weights = [comp["weight"] for comp in diurnal["weekend_profile"]["components"]]
        assert math.isclose(sum(weekend_weights), 1.0, rel_tol=1e-4), (
            f"Weekend circadian mixture weights must sum to 1.0, got {sum(weekend_weights)}"
        )

    def test_adversarial_playbook_relational_integrity(self):
        """Verifies adversarial playbooks reference existing products and valid parameters."""
        prod_data = load_yaml("01_financial_instruments.yaml")
        valid_product_ids = {p["id"] for p in prod_data["card_products"]}
        valid_product_ids.add("ALL_CONSUMER_CREDIT")

        persona_data = load_yaml("02_human_personas.yaml")
        valid_cluster_ids = {c["id"] for c in persona_data["consumer_clusters"]}
        valid_cluster_ids.add("ALL")

        adv_data = load_yaml("04_adversarial_playbooks.yaml")
        playbooks = adv_data["adversarial_playbooks"]

        for pb in playbooks:
            pbid = pb["id"]
            # Verify target products exist if declared
            if "target_products" in pb:
                for t_prod in pb["target_products"]:
                    assert t_prod in valid_product_ids, (
                        f"Playbook {pbid} references unknown product: {t_prod}"
                    )

            # Verify target clusters exist if declared
            if "target_clusters" in pb:
                for t_cluster in pb["target_clusters"]:
                    assert t_cluster in valid_cluster_ids, (
                        f"Playbook {pbid} references unknown cluster: {t_cluster}"
                    )

    def test_payment_rails_plumbing_parameters(self):
        """Verifies AFD holds, dining tip tolerances, and PSD2 SCA exemption bounds."""
        data = load_yaml("03_payment_rail_gaps.yaml")

        # AFD
        afd = data["pre_authorization_protocols"]["automated_fuel_dispenser_mcc_5542"]
        assert afd["standard_auth_hold_cents"] == 17500  # $175.00
        assert afd["fleet_auth_hold_cents"] == 35000     # $350.00

        # Dining tip tolerance
        dining = data["pre_authorization_protocols"]["dining_restaurant_mcc_5812"]
        assert dining["tip_tolerance_percentage"] == 20.0

        # PSD2 Article 16
        art16 = data["emv_3ds_sca_exemptions"]["low_value_article_16"]
        assert art16["max_single_tx_eur"] == 30.00
        assert art16["cumulative_spend_circuit_breaker_eur"] == 100.00
        assert art16["consecutive_tx_circuit_breaker"] == 5

    def test_dispute_and_chargeback_lifecycle_parameters(self):
        """Verifies Visa VCR, CE 3.0, and arbitration economic hurdles."""
        data = load_yaml("03_payment_rail_gaps.yaml")
        assert "global_dispute_and_chargeback_lifecycle" in data
        lifecycle = data["global_dispute_and_chargeback_lifecycle"]

        # Visa VCR Allocation
        vcr_alloc = lifecycle["visa_claims_resolution_vcr"]["allocation_workflow"]
        assert vcr_alloc["primary_reason_code"] == "10.4_OTHER_FRAUD_CARD_ABSENT"
        assert vcr_alloc["second_presentment_allowed"] is False
        assert vcr_alloc["max_filing_window_days"] == 120

        # Mastercard MasterCom
        mc = lifecycle["mastercard_mastercom_workflow"]
        assert mc["primary_fraud_reason_code"] == "4837_NO_CARDHOLDER_AUTHORIZATION"
        assert mc["max_filing_window_days"] == 120

        # Visa CE 3.0 rules
        ce3 = lifecycle["visa_compelling_evidence_3_0"]["qualification_criteria"]
        assert ce3["historical_undisputed_settled_tx_count"] == 2
        assert ce3["historical_window_days"] == [120, 365]
        assert ce3["dispute_free_invariant"] is True
        assert ce3["factor_matching"]["min_distinct_factors"] == 2
        assert "IP_ADDRESS" in ce3["factor_matching"]["mandatory_anchor_factor"]

        # Network arbitration economic veto
        arb = lifecycle["network_arbitration_tollgate"]
        assert arb["filing_fee_usd"] == 500.00
        assert arb["liability_determination"] == "LOSER_PAYS_ALL"
        assert arb["rational_economic_veto_ceiling_usd"] == 1000.00

    def test_merchant_acquirer_risk_and_monitoring_programs(self):
        """Verifies Visa VAMP, Mastercard ECP/EFM, MATCH list, and gateway velocity filters."""
        data = load_yaml("03_payment_rail_gaps.yaml")
        assert "merchant_acquirer_and_gateway_risk" in data
        assert "scheme_monitoring_programs" in data

        # Gateway velocity & geo mismatch
        gw = data["merchant_acquirer_and_gateway_risk"]
        assert gw["pre_auth_gateway_velocity_filters"]["card_to_ip_max_distinct_pans"] == 3
        weights = gw["five_way_geolocation_risk_matrix"]["weights"]
        assert sum(weights.values()) == 100
        assert gw["five_way_geolocation_risk_matrix"]["high_risk_threshold_score"] == 75

        # Visa VAMP 2025/2026
        vamp = data["scheme_monitoring_programs"]["visa_acquirer_monitoring_program_vamp"]
        assert vamp["excessive_merchant_threshold_2025"] == 0.022
        assert vamp["excessive_merchant_threshold_2026"] == 0.015
        assert vamp["min_monthly_volume_events"] == 1500
        assert vamp["non_compliance_fee_per_event_usd"] == 8.00

        # Mastercard ECP
        ecp = data["scheme_monitoring_programs"]["mastercard_excessive_chargeback_program_ecp"]
        assert ecp["tier_1_ecm"]["min_chargebacks"] == 100
        assert ecp["tier_2_hecm"]["min_chargebacks"] == 300
        assert ecp["tier_2_hecm"]["min_chargeback_ratio"] == 0.03

        # Mastercard MATCH
        match_sys = data["scheme_monitoring_programs"]["mastercard_match_system"]
        assert match_sys["mandatory_listing_window_days"] == 5
        assert match_sys["retention_period_years"] == 5
        assert match_sys["total_reason_codes"] == 14

    def test_technical_cybercrime_and_mule_pipelines(self):
        """Verifies PAN enumeration additive guessing complexity, VAAI, and money mule rules."""
        data = load_yaml("04_adversarial_playbooks.yaml")
        playbooks = {pb["id"]: pb for pb in data["adversarial_playbooks"]}

        # Distributed BIN enumeration (PEA)
        assert "ADV_DISTRIBUTED_BIN_ENUMERATION" in playbooks
        pea = playbooks["ADV_DISTRIBUTED_BIN_ENUMERATION"]
        assert pea["phase_1_expiry_probing"]["search_space_combinations"] == 60
        assert pea["phase_1_expiry_probing"]["parallel_merchants_count"] == 30
        assert pea["phase_1_expiry_probing"]["requests_per_merchant"] == 2
        assert pea["phase_2_cvv_probing"]["search_space_combinations"] == 1000
        assert pea["phase_2_cvv_probing"]["parallel_merchants_count"] == 200
        assert pea["phase_2_cvv_probing"]["requests_per_merchant"] == 5
        assert pea["network_countermeasures"]["visa_vaai"]["threshold_decline"] == 75

        # Triangulation Fraud
        assert "ADV_TRIANGULATION_FRAUD" in playbooks
        tri = playbooks["ADV_TRIANGULATION_FRAUD"]
        assert tri["chargeback_multiplier_range"] == [3.75, 4.23]

        # Money Mule Pipeline
        mules = data["money_mule_laundering_pipeline"]
        assert mules["layer_1_ingress"]["dwell_time_seconds"] == [60, 180]
        assert mules["layer_2_smurfing_fan_out"]["fan_out_accounts_count"] == [4, 8]
        assert mules["layer_2_smurfing_fan_out"]["tranche_ceiling_usd"] == 1950.00
        assert mules["layer_3_crypto_severance"]["crypto_asset"] == "USDT_TRC20"

    def test_rbi_dispute_and_compensation_mandates(self):
        """Verifies RBI customer limited liability tiers, TAT compensation, and CFCFRMS."""
        data = load_yaml("05_india_payment_rails.yaml")
        rails = data["india_regulatory_rails"]

        # RBI Circular RBI/2017-18/15
        assert "customer_protection_limiting_liability" in rails
        cp = rails["customer_protection_limiting_liability"]
        assert cp["zero_liability_scenarios"]["third_party_breach_reported_within_working_days"] == 3
        tiers = cp["limited_liability_tiers_reporting_4_to_7_working_days"]
        assert tiers["bsbd_pmjdy_accounts_max_liability_paisa"] == 500000        # ₹5,000
        assert tiers["savings_ppi_credit_cards_le_5_lakh_paisa"] == 1000000      # ₹10,000
        assert tiers["credit_cards_gt_5_lakh_and_corporate_paisa"] == 2500000    # ₹25,000
        assert cp["provisional_shadow_credit_mandate_working_days"] == 10
        assert cp["absolute_resolution_sla_calendar_days"] == 90
        assert cp["burden_of_proof"] == "BANK_BEARS_100_PERCENT_BURDEN"

        # RBI Circular RBI/2019-20/67 (TAT compensation)
        assert "harmonisation_tat_failed_transactions" in rails
        tat = rails["harmonisation_tat_failed_transactions"]
        assert tat["auto_reversal_turnaround_times"]["atm_cash_not_dispensed_calendar_days"] == 5
        assert tat["auto_reversal_turnaround_times"]["upi_imps_beneficiary_not_credited_calendar_days"] == 1
        assert tat["delayed_reversal_statutory_penalty_per_day_paisa"] == 10000  # ₹100.00 INR/day

        # PA Escrow mandate (RBI/DPSS/2025-26/115)
        assert "payment_aggregator_escrow_settlement" in rails
        pa = rails["payment_aggregator_escrow_settlement"]
        assert pa["merchant_payout_settlement_sla"] == "T_PLUS_1_BUSINESS_DAYS_MAX"

        # CFCFRMS / 1930
        assert "cfcfrms_1930_defensive_race" in data
        cfc = data["cfcfrms_1930_defensive_race"]
        assert cfc["helpline_number"] == 1930
        assert cfc["critical_golden_hour_cutoff_seconds"] == 900
        assert cfc["inter_bank_api_lien_propagation_latency_seconds"] == 180


def test_a_custom_spec_directory_without_the_us_targets_loads(tmp_path):
    """load_all_specs(spec_dir=...) decided whether to read the optional US targets by
    looking in the packaged spec directory instead of the one it was given, so a custom
    directory without that file failed to load."""
    import shutil
    from pathlib import Path

    import prism_generator
    from prism_generator.spec_loader import load_all_specs

    packaged = Path(prism_generator.__file__).resolve().parents[1] / "spec"
    for f in packaged.glob("*.yaml"):
        if f.name != "09_us_calibration_targets.yaml":
            shutil.copy(f, tmp_path / f.name)
    registry = load_all_specs(spec_dir=str(tmp_path))
    assert registry is not None
