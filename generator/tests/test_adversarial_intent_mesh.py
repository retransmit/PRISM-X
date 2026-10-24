# tests/test_adversarial_intent_mesh.py
# Automated Deterministic Unit & Falsification Suite for Slice 11
# First-Principles Adversarial Intent & Physical Constraint Mesh

import math
import numpy as np
import pytest

from prism_generator.intent import (
    CredentialDossier,
    CredentialTier,
    AnalyticalBeliefState,
    CandidateAction,
    ConstraintPruner,
    InformationDirectedOptimizer,
    MacroOptionType,
    MultiHopSwitchEngine,
)
from prism_generator.rails import ISO8583Response


# ==============================================================================
# SECTION 1: INVARIANT TESTS (ANTI-ROLLBACK CONTRACT)
# ==============================================================================

class TestAntiRollbackContract:
    """Verifies non-negotiable architectural invariants forbidding heuristic rollbacks."""

    def test_invariant_1_feedback_clean_separation(self):
        """Invariant 1: observe_outcome MUST ONLY mutate belief state, never next_amount."""
        belief = AnalyticalBeliefState(
            card_id="card_test_001",
            p_valid=0.50,
            mu_balance=1200.0,
            sigma_balance=400.0,
            min_balance=0.0,
            max_balance=5000.0,
        )
        optimizer = InformationDirectedOptimizer(seed=42)

        # 1. Provide ISO 51 (Insufficient Funds) for $800.00
        prior_max = belief.max_balance
        optimizer.observe_outcome(belief, amount=800.0, response_code="51")

        # Must strictly truncate upper balance bound: max_balance <= 799.99
        assert belief.max_balance == pytest.approx(799.99, abs=0.01)
        assert belief.max_balance < prior_max
        # Must NOT set any hardcoded next_amount attribute on belief or optimizer
        assert not hasattr(belief, "next_amount")
        assert not hasattr(optimizer, "next_amount")

        # 2. Provide ISO 10 (Partial Approval) for $43.20
        optimizer.observe_outcome(belief, amount=200.0, response_code="10", partial_amount=43.20)
        # Must collapse balance exact bounds to partial_amount
        assert belief.min_balance == pytest.approx(43.20, abs=0.01)
        assert belief.max_balance == pytest.approx(43.20, abs=0.01)

    def test_invariant_2_closed_form_shannon_mutual_information(self):
        """Invariant 2: Mutual Information must strictly match Shannon Entropy formula."""
        optimizer = InformationDirectedOptimizer(seed=42)

        # Edge cases: p = 0.0 or 1.0 has 0 entropy, so MI must be exactly 0.0
        assert optimizer.compute_mutual_information(p_valid=0.0, p_approve=0.5) == 0.0
        assert optimizer.compute_mutual_information(p_valid=1.0, p_approve=0.5) == 0.0

        # Noiseless channel: p = 0.50, p_approve = 1.0
        # If approved -> p'=1.0 (H=0); if decline -> p'=0.0 (H=0)
        # Expected post-entropy = 0.0 => MI = H(0.5) - 0.0 = ln(2)
        mi_noiseless = optimizer.compute_mutual_information(p_valid=0.50, p_approve=1.0)
        assert mi_noiseless == pytest.approx(math.log(2.0), abs=1e-4)

        # Noisy channel: p = 0.50, p_approve = 0.50
        # Decline leaves uncertainty: P(V=1|O=0) = 0.25/0.75 = 1/3 => H(1/3) ~ 0.6365
        # Expected post-entropy = 0.75 * 0.6365 = 0.4773 => MI = ln(2) - 0.4773 ~ 0.21576
        mi_noisy = optimizer.compute_mutual_information(p_valid=0.50, p_approve=0.50)
        assert mi_noisy == pytest.approx(0.21576, abs=1e-4)

    def test_invariant_3_physical_lattice_pruning(self):
        """Invariant 3: Physical constraints must strictly prune impossible channel actions."""
        pruner = ConstraintPruner()

        # Case A: CNP Fullz dossier (has CVV, lacks physical chip and OTP stream)
        fullz_dossier = CredentialDossier(
            tier=CredentialTier.TIER_CNP_FULLZ,
            pan="4111111111111111",
            expiry_month=12,
            expiry_year=2028,
            cvv2="123",
            cardholder_name="Alice Smith",
            billing_zip="94105",
            has_chip_cryptogram=False,
            has_live_otp=False,
        )

        candidates = [
            CandidateAction(mcc=5732, amount=150.0, channel="CP_POS_CHIP"),
            CandidateAction(mcc=5732, amount=150.0, channel="CP_CONTACTLESS_NFC"),
            CandidateAction(mcc=5732, amount=150.0, channel="CNP_WEB"),
            CandidateAction(mcc=5732, amount=150.0, channel="PROVISION_DIGITAL_WALLET"),
        ]

        valid_actions = pruner.prune_candidates(fullz_dossier, candidates, velocity_kmh=0.0)

        # CP_POS_CHIP must be pruned (no chip)
        # CP_CONTACTLESS_NFC must be pruned (no bound token/NFC)
        # PROVISION_DIGITAL_WALLET must be pruned (no live OTP)
        # CNP_WEB must survive
        assert len(valid_actions) == 1
        assert valid_actions[0].channel == "CNP_WEB"

    def test_invariant_3b_kinematic_velocity_pruning(self):
        """Invariant 3b: Supersonic travel velocity (>900 km/h) must be pruned."""
        pruner = ConstraintPruner()
        fullz_dossier = CredentialDossier(
            tier=CredentialTier.TIER_CNP_FULLZ,
            pan="4111111111111111",
            expiry_month=12,
            expiry_year=2028,
            cvv2="123",
        )
        candidate = CandidateAction(mcc=5732, amount=50.0, channel="CNP_WEB")

        # 850 km/h is valid commercial flight
        assert len(pruner.prune_candidates(fullz_dossier, [candidate], velocity_kmh=850.0)) == 1

        # 950 km/h is supersonic / impossible travel
        assert len(pruner.prune_candidates(fullz_dossier, [candidate], velocity_kmh=950.0)) == 0


# ==============================================================================
# SECTION 2: THE 5 MANDATORY FALSIFICATION TESTS
# ==============================================================================

class TestMandatoryFalsifications:
    """The 5 empirical falsification tests that cannot be passed by trivial heuristics."""

    def test_falsification_a1_emergence_of_probing(self):
        """Falsification A1: Probing emerges spontaneously from IDS at high uncertainty."""
        optimizer = InformationDirectedOptimizer(seed=42)

        # High uncertainty: fresh unverified card dossier
        belief = AnalyticalBeliefState(
            card_id="fresh_card_001",
            p_valid=0.40,  # Highly uncertain
            mu_balance=1500.0,
            sigma_balance=500.0,
            min_balance=0.0,
            max_balance=5000.0,
            acquisition_cost_usd=15.0,
        )

        dossier = CredentialDossier(
            tier=CredentialTier.TIER_CNP_FULLZ,
            pan="4147200000000001",
            cvv2="456",
            billing_zip="10001",
        )

        action = optimizer.select_optimal_action(belief, dossier, current_hour_local=14)

        # At p_valid=0.40, mutual information dominates.
        # Action MUST spontaneously choose an information-gathering micro-auth (amount <= $5.00)
        assert action.amount <= 5.0, f"Expected micro-auth probe <= $5.00, got ${action.amount}"
        assert action.macro_option == MacroOptionType.OMEGA_PROBE

    def test_falsification_a2_emergence_of_amount_markdown_on_iso_51(self):
        """Falsification A2: Amount reduction on ISO 51 emerges from bounds, not '* 0.70'."""
        optimizer = InformationDirectedOptimizer(seed=42)

        # Verified card, high balance estimate
        belief = AnalyticalBeliefState(
            card_id="verified_card_002",
            p_valid=1.0,  # Already verified
            mu_balance=2000.0,
            sigma_balance=200.0,
            min_balance=100.0,
            max_balance=5000.0,
        )
        dossier = CredentialDossier(
            tier=CredentialTier.TIER_CNP_FULLZ,
            pan="4147200000000002",
            cvv2="456",
        )

        # Step 1: Agent attempts high-ticket extraction (e.g. $1,200)
        action_1 = optimizer.select_optimal_action(belief, dossier, current_hour_local=14)
        attempt_amount = action_1.amount
        assert attempt_amount >= 500.0

        # Step 2: Issuer responds with ISO 51 (Insufficient Funds)
        optimizer.observe_outcome(belief, amount=attempt_amount, response_code="51")

        # Step 3: Next action MUST be strictly less than previous attempt
        action_2 = optimizer.select_optimal_action(belief, dossier, current_hour_local=14)
        assert action_2.amount < attempt_amount
        assert action_2.amount <= belief.max_balance
        assert action_2.macro_option in (MacroOptionType.OMEGA_BISECT_DRAIN, MacroOptionType.OMEGA_HARVEST)

    def test_falsification_b_physical_invariants_and_telemetry(self):
        """Falsification B: Physical laws and device telemetry cannot be bypassed."""
        multi_hop = MultiHopSwitchEngine()

        # Case 1: Browser tampering (synthetic canvas noise fluctuates)
        result = multi_hop.evaluate_transaction(
            amount=50.0,
            mcc=5732,
            channel="CNP_WEB",
            ip_reputation_score=20,  # clean IP
            canvas_repeatability=False,  # TAMPERED
            tls_fingerprint_match=True,
            velocity_kmh=20.0,
        )
        assert result.response_code == "05"
        assert result.hop_origin == "GATEWAY_FILTER"
        assert "BROWSER_TAMPERING" in result.decline_reason

        # Case 2: Bad IP reputation (MaxMind >= 85)
        result_ip = multi_hop.evaluate_transaction(
            amount=50.0,
            mcc=5732,
            channel="CNP_WEB",
            ip_reputation_score=92,  # DIRTY IP
            canvas_repeatability=True,
            tls_fingerprint_match=True,
            velocity_kmh=20.0,
        )
        assert result_ip.response_code == "05"
        assert result_ip.hop_origin == "GATEWAY_FILTER"

    def test_falsification_c_target_degradation_and_rational_purge(self):
        """Falsification C: Immediate target purge upon ISO 59 / ISO 14; zero wasteful re-attacks."""
        optimizer = InformationDirectedOptimizer(seed=42)
        belief = AnalyticalBeliefState(
            card_id="card_to_burn_003",
            p_valid=0.95,
            mu_balance=1000.0,
            sigma_balance=200.0,
        )

        # Receive ISO 59 (Suspected Fraud - Hard Freeze)
        optimizer.observe_outcome(belief, amount=500.0, response_code="59")

        assert belief.p_valid == 0.0
        assert belief.is_burned is True

        dossier = CredentialDossier(
            tier=CredentialTier.TIER_CNP_FULLZ,
            pan="4147200000000003",
            cvv2="123",
        )

        # Next action MUST be OMEGA_PURGE with 0.0 amount (zero network activity)
        action = optimizer.select_optimal_action(belief, dossier, current_hour_local=14)
        assert action.macro_option == MacroOptionType.OMEGA_PURGE
        assert action.amount == 0.0

    def test_falsification_d_honeypot_policy_inversion(self):
        """Falsification D: Synthetic honeypot inverts bank rules; IDS must invert policy."""
        optimizer = InformationDirectedOptimizer(seed=42)
        belief = AnalyticalBeliefState(
            card_id="honeypot_target_004",
            p_valid=0.50,
            mu_balance=1500.0,
            sigma_balance=300.0,
            min_balance=0.0,
            max_balance=5000.0,
        )
        dossier = CredentialDossier(
            tier=CredentialTier.TIER_CNP_FULLZ,
            pan="4147200000000004",
            cvv2="123",
        )

        # Step 1: Initial probe attempt
        action_1 = optimizer.select_optimal_action(belief, dossier, current_hour_local=14)
        assert action_1.amount <= 5.0

        # Step 2: Honeypot strikes! Bank responds with ISO 59 (Fraud) specifically on micro-probes
        # but the attacker has a syndicate signal that small probes are honeypots (honeypot flag set)
        belief.probe_penalized_by_issuer = True
        optimizer.observe_outcome(belief, amount=action_1.amount, response_code="05")

        # Step 3: An agent with hardcoded probing rules would probe again or fail.
        # An IDS agent inverts its policy, abandoning micro-probes and testing moderate spend (>= $15.0)
        action_2 = optimizer.select_optimal_action(belief, dossier, current_hour_local=14)
        assert action_2.amount >= 15.0, "IDS failed to invert policy away from honeypot micro-probes!"

    def test_falsification_e_darknet_price_elasticity_phase_transition(self):
        """Falsification E: Sunk cost elasticity shifts behavior from smash-and-grab to incubation."""
        optimizer = InformationDirectedOptimizer(seed=42)

        # Scenario 1: Cheap disposable bulk dump ($0.50 acquisition cost)
        cheap_belief = AnalyticalBeliefState(
            card_id="cheap_card_005",
            p_valid=0.35,
            acquisition_cost_usd=0.50,
            mu_balance=500.0,
            sigma_balance=200.0,
        )
        cheap_dossier = CredentialDossier(
            tier=CredentialTier.TIER_CNP_FULLZ,
            pan="4147200000000005",
            cvv2="123",
        )
        # At day time (hour=14), cheap card accepts high alert hazard for immediate extraction
        cheap_action = optimizer.select_optimal_action(cheap_belief, cheap_dossier, current_hour_local=14)
        assert cheap_action.macro_option != MacroOptionType.OMEGA_INCUBATE

        # Scenario 2: High-Net-Worth luxury fullz ($150.00 acquisition cost)
        expensive_belief = AnalyticalBeliefState(
            card_id="expensive_card_006",
            p_valid=0.85,
            acquisition_cost_usd=150.00,  # Heavy sunk cost!
            mu_balance=8000.0,
            sigma_balance=1000.0,
        )
        expensive_dossier = CredentialDossier(
            tier=CredentialTier.TIER_CNP_FULLZ,
            pan="4147200000000006",
            cvv2="123",
        )
        # At day time (hour=14), expensive card cannot risk victim alert hazard, so it schedules incubation until night
        expensive_action = optimizer.select_optimal_action(expensive_belief, expensive_dossier, current_hour_local=14)
        assert expensive_action.macro_option == MacroOptionType.OMEGA_INCUBATE
        assert expensive_action.incubation_seconds > 0


# ==============================================================================
# SECTION 3: MULTI-HOP SWITCH & TOKEN PROVISIONING
# ==============================================================================

class TestMultiHopSwitchAndTokens:
    """Verifies the 4-hop routing pipeline and digital wallet tokenization mechanics."""

    def test_visa_vaai_network_switch_drop(self):
        """Hop 2: High VAAI enumeration score drops transaction before reaching issuer."""
        multi_hop = MultiHopSwitchEngine()

        result = multi_hop.evaluate_transaction(
            amount=1.50,
            mcc=8398,
            channel="CNP_WEB",
            ip_reputation_score=10,
            canvas_repeatability=True,
            tls_fingerprint_match=True,
            velocity_kmh=0.0,
            vaai_score=85,  # Above 75 threshold
        )
        assert result.response_code == "05"
        assert result.hop_origin == "NETWORK_SWITCH_VAAI"

    def test_emvco_token_provisioning_yellow_path(self):
        """EMVCo Token Provisioning (Yellow Path): phished OTP binds card to Apple Pay DPAN."""
        otp_dossier = CredentialDossier(
            tier=CredentialTier.TIER_PHISHED_OTP_STREAM,
            pan="4147200000000007",
            cvv2="888",
            live_otp="749201",
            otp_expiry_seconds=180,
        )

        multi_hop = MultiHopSwitchEngine()
        binding_result = multi_hop.provision_digital_wallet(otp_dossier, submitted_otp="749201")
        assert binding_result.is_provisioned is True
        assert binding_result.dpan is not None
        assert binding_result.token_requestor_id == "APPLE_PAY_TRID"
        assert binding_result.liability_shift == "ISSUER_100_PERCENT"

    def test_adaptive_agent_select_intent_action(self):
        """Verifies AdaptiveFraudsterAgent seamlessly generates intent-driven attacks."""
        from prism_generator.agents import AdaptiveFraudsterAgent, CardholderProfile

        rng = np.random.default_rng(123)
        agent = AdaptiveFraudsterAgent(rng=rng)
        card = CardholderProfile(
            card_id="card_intent_test_999",
            home_lat=37.77,
            home_lon=-122.41,
            work_lat=37.78,
            work_lon=-122.40,
            is_commuter=True,
            credit_limit=5000.0,
            current_balance=200.0,
        )

        # 1. Generate intent action
        attack = agent.select_intent_action(
            card=card,
            sim_time_seconds=14 * 3600.0,  # 14:00 daytime
            world_center_lat=37.77,
            world_center_lon=-122.41,
        )
        assert attack["is_fraud"] == 1
        assert "INTENT_" in attack["scenario_tag"]
        assert attack["macro_option"] in ("OMEGA_PROBE", "OMEGA_HARVEST", "OMEGA_BISECT_DRAIN")

        # 2. Provide feedback ISO 59 (Suspected Fraud)
        agent.receive_feedback(
            response_code=ISO8583Response.SUSPECTED_FRAUD_59,
            trans_status_3ds=None,
            sim_time_seconds=14 * 3600.0 + 30.0,
            card_id=card.card_id,
        )
        # Belief state immediately marks card burned under first-principles intent
        belief = agent.belief_states[card.card_id]
        assert belief.is_burned is True

        # Second decline burns target state in legacy tracker
        agent.receive_feedback(
            response_code=ISO8583Response.SUSPECTED_FRAUD_59,
            trans_status_3ds=None,
            sim_time_seconds=14 * 3600.0 + 60.0,
            card_id=card.card_id,
        )
        assert agent.is_card_burned(card.card_id) is True

        # 3. Next attack MUST be OMEGA_PURGE with amount 0.0
        attack_burned = agent.select_intent_action(
            card=card,
            sim_time_seconds=14 * 3600.0 + 90.0,
            world_center_lat=37.77,
            world_center_lon=-122.41,
        )
        assert attack_burned["macro_option"] == "OMEGA_PURGE"
        assert attack_burned["amount"] == 0.0
