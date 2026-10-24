"""Feature Risk Attribution & Causal Ground-Truth Engine.

Provides:
1. Structural causal risk scoring combining linear feature shifts, logarithmic saturation,
   and multi-feature interaction synergies with bidirectional weights (cryptographic mitigators vs risk signals).
2. Grounded EMV 4.3 Bit 55 (Tag 9F26 ARQC, Tag 9F34 PIN, Tag 95 TVR) and 3DS 2.x (ECI 05/06, CAVV)
   mitigating evidence that reliably protects legitimate anomalous transactions (e.g. Dhanteras gold, overseas travel).
3. Exact Game-Theoretic Feature Attributions:
   - Logit space: Owen multilinear Shapley value decomposition: sum(phi_i^logit) == logit_z - base_logit
   - Probability space: 128-point Gauss-Legendre numerical path integration (Aumann-Shapley / Integrated Gradients)
     with normalized sum matching total probability delta: sum(phi_i^prob) == risk_score - base_risk
4. Pearl's 3-Step Structural Counterfactual Derivation:
   - Abduction: Infer cardholder's baseline latent state X^0 from 30-day Welford historical moments.
   - Action: Intervene with do(A = Normative).
   - Prediction: Synthesize counterfactual transaction and compute input-space deltas.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .experimental.invertible_flow import (
    ConditionalRealNVPFlow,
    LatentAumannShapleyAttributor,
    TransactionFlowFeatureCodec,
)


class BankModel(ABC):
    """Abstract interface for pluggable fraud risk decision models.
    
    Allows external machine learning models (LightGBM, XGBoost, Neural Nets)
    or custom rule engines to be plugged directly into the Prism Generator simulation
    as first-class authorizers.
    """

    @abstractmethod
    def score(self, record: Dict[str, Any]) -> float:
        """Returns continuous risk probability in [0.0, 1.0]."""
        ...

    @abstractmethod
    def decide(self, record: Dict[str, Any], threshold: float = 0.50) -> str:
        """Returns ISO 8583 response code (e.g. '00' Approved, '05' Do Not Honor, '59' Suspected Fraud)."""
        ...


@dataclass
class CausalGroundTruth:
    """Rigorous ground-truth risk attribution and contrastive explanation for a transaction."""
    is_fraud: int
    scenario_tag: str
    risk_score: float
    base_risk: float
    logit_z: float
    base_logit: float
    analytical_shapley_log_odds: Dict[str, float] = field(default_factory=dict)
    analytical_shapley_probability: Dict[str, float] = field(default_factory=dict)
    counterfactual_input_deltas: Dict[str, float] = field(default_factory=dict)
    dominant_causal_driver: str = "baseline"
    active_causal_parents: List[str] = field(default_factory=list)
    counterfactual_mode: str = "CONTRASTIVE_PROFILE_FOIL"
    counterfactual_twin: Dict[str, Any] = field(default_factory=dict)
    normative_baseline: Dict[str, Any] = field(default_factory=dict)
    explanation_narrative: str = ""
    latent_shapley_attributions: Dict[str, float] = field(default_factory=dict)
    pearlian_consistency_error: float = 0.0

    def __post_init__(self) -> None:
        if not self.normative_baseline and self.counterfactual_twin:
            self.normative_baseline = self.counterfactual_twin
        elif not self.counterfactual_twin and self.normative_baseline:
            self.counterfactual_twin = self.normative_baseline


class HeuristicBankScorer(BankModel):
    """Heuristic bank risk scoring function parameterized by domain-expert log-odds weights and synergy kernels.
    
    Serves as a baseline bank decision model and heuristic benchmark, NOT an empirical causal oracle.
    Weights reflect empirical payment fraud heuristics (e.g. CVV mismatches, excessive velocity,
    and EMV cryptogram dampening).
    """

    def __init__(self, base_prevalence: float = 0.0020, seed: int = 42):
        self.base_prevalence = max(1e-5, min(0.999, base_prevalence))
        self.base_logit = math.log(self.base_prevalence / (1.0 - self.base_prevalence))
        self.seed = seed

        # Baseline expected values for normal transactions E[X_i]
        self.feature_baselines: Dict[str, float] = {
            "haversine_velocity_kph": 15.0,
            "amount_to_mean_ratio_30d": 1.0,
            "tx_count_1h": 0.3,
            "tx_count_24h": 2.5,
            "tx_amount_sum_24h_ratio": 0.25,
            "ip_distance_from_home_km": 8.0,
            "is_cross_border_tx": 0.02,
            "avs_mismatch_flag": 0.08,
            "billing_shipping_mismatch": 0.08,
            "cvv_mismatch_flag": 0.01,
            "is_high_risk_mcc": 0.04,
            "is_night_tx": 0.05,
            "emv_arqc_verified": 0.0,
            "emv_pin_verified": 0.0,
            "emv_tvr_clean": 0.0,
            "three_ds_authenticated": 0.0,
            "three_ds_attempted": 0.0,
            "is_carding_probe": 0.0,
        }

        # Structural causal DAG weights (log-odds impact per unit deviation)
        # Bidirectional: negative for cryptographic mitigators, positive for risk signals
        self.structural_weights: Dict[str, float] = {
            # Mitigating Evidence (EMV Bit 55 & 3DS 2.x)
            "emv_arqc_verified": -3.80,
            "emv_pin_verified": -1.80,
            "emv_tvr_clean": -0.80,
            "three_ds_authenticated": -4.20,
            "three_ds_attempted": -1.20,
            # Aggravating Risk Signals
            "haversine_velocity_kph": 0.0055,
            "amount_to_mean_ratio_30d": 0.65,
            "tx_count_1h": 0.85,
            "tx_count_24h": 0.20,
            "tx_amount_sum_24h_ratio": 1.10,
            "ip_distance_from_home_km": 0.0018,
            "is_cross_border_tx": 1.75,
            "avs_mismatch_flag": 1.95,
            "billing_shipping_mismatch": 1.80,
            "cvv_mismatch_flag": 2.80,
            "is_high_risk_mcc": 1.40,
            "is_night_tx": 0.90,
            "is_carding_probe": 2.20,
        }

        # Grounded multi-feature synergy interaction kernels
        self.pairwise_synergies: List[Tuple[str, str, float]] = [
            # Cryptographic dampening synergies
            ("emv_arqc_verified", "amount_to_mean_ratio_30d", -0.50),
            ("emv_arqc_verified", "is_high_risk_mcc", -0.80),
            ("emv_arqc_verified", "haversine_velocity_kph", -0.65),
            # Aggravating risk synergies
            ("haversine_velocity_kph", "ip_distance_from_home_km", 0.00004),
            ("is_night_tx", "amount_to_mean_ratio_30d", 0.45),
            ("is_night_tx", "is_cross_border_tx", 1.20),
            ("amount_to_mean_ratio_30d", "is_high_risk_mcc", 0.55),
            ("tx_count_1h", "tx_amount_sum_24h_ratio", 0.75),
            ("avs_mismatch_flag", "cvv_mismatch_flag", 1.85),
            ("is_carding_probe", "avs_mismatch_flag", 1.25),
        ]

        self.three_way_synergies: List[Tuple[str, str, str, float]] = [
            ("is_night_tx", "is_cross_border_tx", "amount_to_mean_ratio_30d", 0.35),
            ("is_carding_probe", "avs_mismatch_flag", "cvv_mismatch_flag", 1.50),
        ]

        # 128-point Gauss-Legendre quadrature nodes and weights for exact Aumann-Shapley path attribution
        x_nodes, w_weights = np.polynomial.legendre.leggauss(128)
        self._gl_t = 0.5 * (x_nodes + 1.0)
        self._gl_wt = 0.5 * w_weights

        # Invertible Normalizing Flow for exact Pearlian abduction & on-manifold counterfactuals
        self.flow = ConditionalRealNVPFlow(seed=self.seed)
        self.codec = TransactionFlowFeatureCodec
        self.latent_attributor = LatentAumannShapleyAttributor(self.flow)

        # Round-tripping x -> u -> x on every record is a self-check of the flow's algebraic
        # inverse, not a quantity any export consumes. It is on by default (tests and single
        # evaluations rely on `pearlian_consistency_error`), and the batch generator turns it
        # off, where it cost about 8% of per-record evaluation time to re-verify a property the
        # architecture guarantees and tests/test_invertible_dscm.py asserts.
        self.verify_flow_consistency: bool = True

    def _logistic(self, z: float) -> float:
        """Numerically stable standard sigmoid."""
        if z >= 35.0:
            return 1.0
        elif z <= -35.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def _flow_scorer(self, x_vec: np.ndarray, mean_30d: float) -> float:
        """Scalar risk score used as the target of the Aumann-Shapley path integral.

        A monotone function of amount saturation against the card's 30-day mean and of
        velocity, both read out of the flow's feature space.
        """
        v_kph = min(900.0, max(0.0, float(x_vec[1])))
        log_amt = min(15.0, max(-5.0, float(x_vec[0])))
        amt = math.exp(log_amt)
        ratio = amt / max(mean_30d, 1.0)
        sat_ratio = math.log(1.0 + max(0.0, ratio - 1.0))
        z = self.base_logit + 0.65 * sat_ratio + 0.0055 * v_kph
        return self._logistic(z)

    def _flow_scorer_batch(self, x_rows: np.ndarray, mean_30d: float) -> np.ndarray:
        """`_flow_scorer` for a stack of rows: element-wise equivalents of every operation
        above, so path integration scores n_steps x (1 + dim) rows in one call instead of
        that many interpreter calls. Both are exercised against each other in
        tests/test_synthesizer/test_causal_shapley.py."""
        v_kph = np.clip(x_rows[:, 1], 0.0, 900.0)
        log_amt = np.clip(x_rows[:, 0], -5.0, 15.0)
        ratio = np.exp(log_amt) / max(mean_30d, 1.0)
        sat_ratio = np.log1p(np.maximum(0.0, ratio - 1.0))
        z = self.base_logit + 0.65 * sat_ratio + 0.0055 * v_kph
        out = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        return np.where(z <= -35.0, 0.0, out)

    def _authentication_deltas(self, record: Dict[str, Any]) -> Dict[str, float]:
        """The five EMV/3DS authentication deviations for a record.

        Engine-produced records carry all five: rails authenticate *before* the issuer's
        authorization decision, so the causal evaluation consumes the same outcome the
        authorization used. A record without them (hand-built fixtures, external callers)
        falls back to the terminal's cryptographic capability as read from the channel or
        POS entry mode. The fallback is deliberately not `is_fraud`: reading the label here
        made all five attributes a perfect oracle, which spec/07 forbids and which made the
        pre-authentication risk score depend on the authentication result it decides.
        """
        channel = str(record.get("channel_type", ""))
        chip_capable = channel in ("CP_POS_CHIP", "CP_POS_CONTACTLESS", "CP_CONTACTLESS_NFC")
        pos_entry = str(record.get("pos_entry_mode", ""))
        if pos_entry:
            chip_capable = pos_entry in ("051", "071")
        status_3ds = str(record.get("trans_status_3ds", ""))

        def flag(name: str, default: bool) -> float:
            if name in record:
                return 1.0 if bool(record[name]) else 0.0
            return 1.0 if default else 0.0

        arqc_verified = flag("emv_arqc_verified", chip_capable)
        return {
            "emv_arqc_verified": arqc_verified,
            # The rail signals PIN entry through intent.pin_entered, which the engine never
            # sets, so PIN-verified is 1.0 only when a record states it (see docs/PROJECT_CONTEXT.md 7.1).
            "emv_pin_verified": flag("emv_pin_verified", False),
            "emv_tvr_clean": flag("emv_tvr_clean", arqc_verified == 1.0),
            "three_ds_authenticated": flag("three_ds_authenticated", status_3ds in ("Y", "C")),
            "three_ds_attempted": flag("three_ds_attempted", status_3ds in ("A", "C")),
        }

    def _structural_state(
        self,
        record: Dict[str, Any],
        scenario_tag: str,
        authentication: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Builds the structural deviation vector and the scalars `evaluate` reuses later.

        Returns `(deltas, state)`. `deltas` is the flow's x-vector equivalent in the SCM's
        feature space; `state` carries the handful of raw quantities (amount, velocity,
        distance, credential mismatches) that the counterfactual baseline construction needs,
        so they are derived exactly once.
        """
        amount = float(record.get("amount", 25.0))
        mean_30d = float(record.get("user_avg_tx_amount_30d", 25.0))
        ratio_30d = amount / max(mean_30d, 1.0)

        velocity_kph = float(record.get("haversine_velocity_kph", 0.0))
        channel = str(record.get("channel_type", ""))

        if not channel.startswith("CP") and scenario_tag != "COUNTERFEIT_CLONE" and scenario_tag != "IMPOSSIBLE_TRAVEL":
            effective_velocity = 0.0
        else:
            effective_velocity = velocity_kph

        count_1h = float(record.get("tx_count_1h", 0))
        count_24h = float(record.get("tx_count_24h", 0))
        sum_24h = float(record.get("tx_amount_sum_24h", 0.0))
        credit_limit = float(record.get("credit_limit", 5000.0))
        sum_ratio = sum_24h / max(credit_limit, 500.0)

        ip_dist = float(record.get("ip_distance_from_home_km", 0.0))
        cross_border = 1.0 if bool(record.get("is_cross_border", False)) else 0.0

        avs_code = str(record.get("avs_match_code", "Y"))
        avs_mismatch = 1.0 if avs_code in ("N", "U") else 0.0

        bill_ship = int(record.get("billing_shipping_match", 1))
        bill_mismatch = 1.0 if bill_ship == 0 else 0.0

        cvv_flag = int(record.get("cvv_match_flag", 1))
        cvv_mismatch = 1.0 if cvv_flag == 0 else 0.0

        mcc = int(record.get("mcc", 5411))
        is_high_risk = 1.0 if mcc in (5094, 6051, 5732, 7995, 8398) else 0.0

        hour = int(record.get("hour_of_day", 14))
        is_night = 1.0 if (1 <= hour <= 5) else 0.0

        # Cryptographic & Authentication Mitigators (from the rail's recorded outcome)
        arqc_verified = authentication["emv_arqc_verified"]
        three_ds_auth = authentication["three_ds_authenticated"]

        # Carding Probe Pattern
        is_carding = 1.0 if (channel.startswith("CNP") and amount <= 3.00 and count_1h >= 2 and three_ds_auth == 0.0) else 0.0

        # Logarithmic saturation prevents massive high-ticket legitimate spikes (e.g. Dhanteras gold, weddings)
        # from linearly blowing up log-odds beyond cryptographic mitigating evidence.
        saturated_amount_ratio = math.log(1.0 + max(0.0, ratio_30d - 1.0))

        deltas: Dict[str, float] = {
            "haversine_velocity_kph": effective_velocity,
            "amount_to_mean_ratio_30d": saturated_amount_ratio,
            "tx_count_1h": count_1h,
            "tx_count_24h": count_24h,
            "tx_amount_sum_24h_ratio": sum_ratio,
            "ip_distance_from_home_km": ip_dist,
            "is_cross_border_tx": cross_border,
            "avs_mismatch_flag": avs_mismatch,
            "billing_shipping_mismatch": bill_mismatch,
            "cvv_mismatch_flag": cvv_mismatch,
            "is_high_risk_mcc": is_high_risk,
            "is_night_tx": is_night,
            "emv_arqc_verified": arqc_verified,
            "emv_pin_verified": authentication["emv_pin_verified"],
            "emv_tvr_clean": authentication["emv_tvr_clean"],
            "three_ds_authenticated": three_ds_auth,
            "three_ds_attempted": authentication["three_ds_attempted"],
            "is_carding_probe": is_carding,
        }
        state = {
            "amount": amount,
            "mean_30d": mean_30d,
            "effective_velocity": effective_velocity,
            "ip_dist": ip_dist,
            "avs_mismatch": avs_mismatch,
            "bill_mismatch": bill_mismatch,
            "cvv_mismatch": cvv_mismatch,
        }
        return deltas, state

    def _attribution_components(
        self, deltas: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float, float, float]:
        """Linear, pairwise and three-way structural components plus their sums.

        Returns `(A_0, A_1, A_2, c1, c2, c3)`. Shared by the full evaluation and by the
        pre-authentication score so the interaction kernels exist once.
        """
        # Linear component A_{i,0}
        A_0 = {
            feat_name: self.structural_weights[feat_name] * deltas[feat_name]
            for feat_name in self.structural_weights
        }
        c1 = sum(A_0.values())

        # Pairwise interaction components A_{i,1}
        A_1 = {feat_name: 0.0 for feat_name in self.structural_weights}
        c2 = 0.0
        for f1, f2, weight in self.pairwise_synergies:
            # When in-person EMV Chip cryptogram is physically authenticated, POS network velocity
            # and IP distance synergy is physically suppressed (card is present at physical terminal).
            if (f1, f2) == ("haversine_velocity_kph", "ip_distance_from_home_km") and deltas["emv_arqc_verified"] == 1.0:
                continue
            term = weight * deltas[f1] * deltas[f2]
            if abs(term) > 1e-12:
                A_1[f1] += term
                A_1[f2] += term
                c2 += term

        # 3-way interaction components A_{i,2}
        A_2 = {feat_name: 0.0 for feat_name in self.structural_weights}
        c3 = 0.0
        for f1, f2, f3, weight in self.three_way_synergies:
            term = weight * deltas[f1] * deltas[f2] * deltas[f3]
            if abs(term) > 1e-12:
                A_2[f1] += term
                A_2[f2] += term
                A_2[f3] += term
                c3 += term

        return A_0, A_1, A_2, c1, c2, c3

    def probability_on_columns(
        self,
        record: Dict[str, Any],
        scenario_tag: str,
        column_values: Dict[str, Any],
    ) -> np.ndarray:
        """Approval probability as a function of the spec/21 explained columns.

        `column_values` maps explained-column names (the vocabulary of
        `causal_contract.feature_names()`) to scalars or equal-length arrays. Every
        field the columns do not name -- channel, MCC, hour of day, the card's 30-day
        mean ticket, credit limit, the non-mapped authentication flags -- is held at
        the record's own value, read exactly as `evaluate` reads it. Returns the
        approval probability per column vector, elementwise.

        Why this exists: the Pillar-4 protocol grades a practical black-box explainer
        against the exact attribution of the SAME function, so both sides must be able
        to evaluate the scorer at arbitrary points of the explained-column space (the
        exact side enumerates all 2^11 coalition points per record). This method is a
        vectorised transcription of `_structural_state` + `_attribution_components`;
        the benchmark runner asserts the observed point against the record's stored
        `risk_score`, so the transcription cannot drift from `evaluate` unnoticed.
        """
        authentication = self._authentication_deltas(record)
        deltas, state = self._structural_state(record, scenario_tag, authentication)

        arrays = [np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in column_values.values()]
        size = max((a.size for a in arrays), default=1)

        def _column(name: str, default: float) -> np.ndarray:
            value = column_values.get(name)
            if value is None:
                return np.full(size, float(default))
            return np.broadcast_to(np.atleast_1d(np.asarray(value, dtype=np.float64)), (size,)).astype(np.float64)

        channel = str(record.get("channel_type", ""))
        raw_amount = _column("amount", state["amount"])
        mean_30d = max(state["mean_30d"], 1.0)

        # The scorer's feature space, with the eleven columns overridden. Derivations
        # mirror `_structural_state` line for line (same saturations, same gates).
        features: Dict[str, np.ndarray] = {}
        features["amount_to_mean_ratio_30d"] = np.log1p(
            np.maximum(0.0, raw_amount / mean_30d - 1.0)
        )
        if channel.startswith("CP") or scenario_tag in ("COUNTERFEIT_CLONE", "IMPOSSIBLE_TRAVEL"):
            features["haversine_velocity_kph"] = _column("haversine_velocity_kph", state["effective_velocity"])
        else:
            features["haversine_velocity_kph"] = np.zeros(size)
        features["ip_distance_from_home_km"] = _column("ip_distance_from_home_km", state["ip_dist"])
        features["tx_count_1h"] = _column("tx_count_1h", deltas["tx_count_1h"])
        features["tx_count_24h"] = _column("tx_count_24h", deltas["tx_count_24h"])
        features["tx_amount_sum_24h_ratio"] = np.full(size, deltas["tx_amount_sum_24h_ratio"])
        features["is_cross_border_tx"] = _column("is_cross_border", deltas["is_cross_border_tx"])
        features["avs_mismatch_flag"] = _column("avs_mismatch_flag", state["avs_mismatch"])
        features["billing_shipping_mismatch"] = _column("billing_shipping_mismatch", state["bill_mismatch"])
        features["cvv_mismatch_flag"] = _column("cvv_mismatch_flag", state["cvv_mismatch"])
        features["is_high_risk_mcc"] = np.full(size, deltas["is_high_risk_mcc"])
        features["is_night_tx"] = np.full(size, deltas["is_night_tx"])
        features["emv_arqc_verified"] = _column("emv_arqc_verified", deltas["emv_arqc_verified"])
        features["emv_pin_verified"] = np.full(size, deltas["emv_pin_verified"])
        features["emv_tvr_clean"] = np.full(size, deltas["emv_tvr_clean"])
        features["three_ds_authenticated"] = _column("three_ds_authenticated", deltas["three_ds_authenticated"])
        features["three_ds_attempted"] = np.full(size, deltas["three_ds_attempted"])
        # The carding-probe pattern is a function of explained columns (amount,
        # count_1h, 3DS) plus the record's channel, so it must follow the overrides.
        features["is_carding_probe"] = np.where(
            (raw_amount <= 3.00) & (features["tx_count_1h"] >= 2)
            & (features["three_ds_authenticated"] == 0.0) & channel.startswith("CNP"),
            1.0,
            0.0,
        )

        z = np.full(size, self.base_logit)
        for feat_name, weight in self.structural_weights.items():
            z = z + weight * features[feat_name]
        for f1, f2, weight in self.pairwise_synergies:
            term = weight * features[f1] * features[f2]
            if (f1, f2) == ("haversine_velocity_kph", "ip_distance_from_home_km"):
                # Same physical suppression as `_attribution_components`: a present,
                # chip-authenticated card has no network-velocity/IP synergy.
                term = np.where(features["emv_arqc_verified"] == 1.0, 0.0, term)
            z = z + term
        for f1, f2, f3, weight in self.three_way_synergies:
            z = z + weight * features[f1] * features[f2] * features[f3]

        # Elementwise twin of `_logistic`, including its saturation bounds.
        saturated = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        return np.where(z >= 35.0, 1.0, np.where(z <= -35.0, 0.0, saturated))

    def pre_authentication_score(
        self,
        record: Dict[str, Any],
        scenario_tag: str = "ORGANIC_NORMAL",
    ) -> float:
        """Risk score from the signals that exist before authentication is attempted.

        This is the score the ACS/issuer uses to decide the 3DS exemption/challenge and the
        EMV outcome, which is why it must not include those outcomes: the engine computes it
        first, feeds it to `rails.RailSwitch.authenticate()`, attaches the result to the
        record and only then calls `evaluate()` for the published score and attributions.
        Structurally it is `evaluate`'s logit with the five authentication deviations held
        at their unauthenticated value (0), i.e. the same model, not a second one.
        """
        authentication = {name: 0.0 for name in (
            "emv_arqc_verified",
            "emv_pin_verified",
            "emv_tvr_clean",
            "three_ds_authenticated",
            "three_ds_attempted",
        )}
        deltas, _state = self._structural_state(record, scenario_tag, authentication)
        _A_0, _A_1, _A_2, c1, c2, c3 = self._attribution_components(deltas)
        return self._logistic(self.base_logit + c1 + c2 + c3)

    def evaluate(
        self,
        record: Dict[str, Any],
        scenario_tag: str = "ORGANIC_NORMAL",
        factual_counterfactual: Optional[Dict[str, Any]] = None,
    ) -> CausalGroundTruth:
        """Evaluates causal risk and constructs exact counterfactual twin and attributions."""
        authentication = self._authentication_deltas(record)
        deltas, state = self._structural_state(record, scenario_tag, authentication)
        amount = state["amount"]
        mean_30d = state["mean_30d"]
        effective_velocity = state["effective_velocity"]
        ip_dist = state["ip_dist"]
        avs_mismatch = state["avs_mismatch"]
        bill_mismatch = state["bill_mismatch"]
        cvv_mismatch = state["cvv_mismatch"]
        is_fraud = int(record.get("is_fraud", 0))

        # 1. Compute Logit-Space Feature Attributions via Owen Multilinear Formula
        A_0, A_1, A_2, c1, c2, c3 = self._attribution_components(deltas)

        # Feature attributions in logit space: phi_i^logit = A_{i,0} + 1/2 A_{i,1} + 1/3 A_{i,2}
        raw_phi_logit = {
            feat_name: A_0[feat_name] + 0.5 * A_1[feat_name] + (1.0 / 3.0) * A_2[feat_name]
            for feat_name in self.structural_weights
        }
        total_logit_shift = c1 + c2 + c3
        logit_z = self.base_logit + total_logit_shift
        risk_score = self._logistic(logit_z)
        base_risk = self.base_prevalence

        # Clean unrounded or minimally rounded logit attributions with exact mathematical efficiency: sum == total_logit_shift
        shapley_logit = {k: round(v, 6) for k, v in raw_phi_logit.items()}

        # 2. Probability Space Attributions (Numerical Path Integration via 128-Point Quadrature)
        # Along straight-line path x(t) = x_0 + t * delta_x:
        # z(t) = base_logit + c1 * t + c2 * t^2 + c3 * t^3
        # phi_i^AS = A_{i,0} M0 + A_{i,1} M1 + A_{i,2} M2
        # where Mm = int_0^1 t^m sigma'(z(t)) dt
        z_t = self.base_logit + c1 * self._gl_t + c2 * (self._gl_t ** 2) + c3 * (self._gl_t ** 3)
        sig = 1.0 / (1.0 + np.exp(-np.clip(z_t, -35.0, 35.0)))
        sig_prime = sig * (1.0 - sig)

        M0 = float(np.sum(self._gl_wt * sig_prime))
        M1 = float(np.sum(self._gl_wt * self._gl_t * sig_prime))
        M2 = float(np.sum(self._gl_wt * (self._gl_t ** 2) * sig_prime))

        # Path-integrated attributions (normalized to ensure sum_i phi_i == delta_p)
        delta_p = risk_score - base_risk
        raw_psi = {
            feat_name: A_0[feat_name] * M0 + A_1[feat_name] * M1 + A_2[feat_name] * M2
            for feat_name in self.structural_weights
        }
        sum_psi = sum(raw_psi.values())
        if abs(sum_psi) > 1e-12 and abs(delta_p) > 1e-12:
            scale_factor = delta_p / sum_psi
            shapley_prob = {feat_name: round(v * scale_factor, 6) for feat_name, v in raw_psi.items()}
        elif abs(delta_p) <= 1e-12:
            shapley_prob = {feat_name: 0.0 for feat_name in self.structural_weights}
        else:
            shapley_prob = {feat_name: round(v, 6) for feat_name, v in raw_psi.items()}

        # 3. Grounded Normative Baseline & Contrastive Attribution Construction
        cf_input_deltas: Dict[str, float] = {}

        # Invertible Flow Abduction-Action-Prediction Pipeline
        x_obs, C_obs = self.codec.encode(record)
        C_norm = C_obs.copy()
        C_norm[0] = 0.0  # Surgical intervention: do(is_fraud = 0)
        u_star = self.flow.abduce(x_obs, C_obs)
        x_cf = self.flow.predict_counterfactual(u_star, C_norm)
        if self.verify_flow_consistency:
            consistency_error = self.flow.verify_pearlian_consistency(x_obs, C_obs)
        else:
            # NaN means "not measured on this path", which is not the same as a
            # round-trip error of zero; see the flag's definition in __init__.
            consistency_error = float("nan")

        # On-manifold latent Aumann-Shapley path integration
        # Evaluated for fraud, anomalous scenarios, or counterfactual foils to maintain high simulation TPS
        if is_fraud == 1 or scenario_tag != "ORGANIC_NORMAL":
            latent_shapley = self.latent_attributor.attribute(
                x_obs=x_obs,
                context=C_obs,
                scorer_fn=lambda v: self._flow_scorer(v, mean_30d),
                u_obs=u_star,
                scorer_batch_fn=lambda X: self._flow_scorer_batch(X, mean_30d),
            )
        else:
            latent_shapley = {}

        if factual_counterfactual is not None:
            normative_baseline = dict(factual_counterfactual)
            baseline_amount = float(factual_counterfactual.get("amount", mean_30d))
            baseline_ip = float(factual_counterfactual.get("ip_distance_from_home_km", 0.0))
            baseline_vel = float(factual_counterfactual.get("haversine_velocity_kph", 0.0))
            baseline_bill = 1.0 if int(factual_counterfactual.get("billing_shipping_match", 1)) == 0 else 0.0
            baseline_avs = 1.0 if str(factual_counterfactual.get("avs_match_code", "Y")) in ("N", "U") else 0.0
            baseline_cvv = 1.0 if int(factual_counterfactual.get("cvv_match_flag", 1)) == 0 else 0.0
            counterfactual_mode = "TRANSACTIONAL_MUTATION" if is_fraud == 1 else "ORGANIC_BASELINE"

            cf_input_deltas["amount"] = round(amount - baseline_amount, 2)
            cf_input_deltas["haversine_velocity_kph"] = round(effective_velocity - baseline_vel, 2)
            cf_input_deltas["ip_distance_from_home_km"] = round(ip_dist - baseline_ip, 2)
            cf_input_deltas["billing_shipping_mismatch"] = float(bill_mismatch - baseline_bill)
            cf_input_deltas["avs_mismatch_flag"] = float(avs_mismatch - baseline_avs)
            cf_input_deltas["cvv_mismatch_flag"] = float(cvv_mismatch - baseline_cvv)
        else:
            normative_baseline = dict(record)
            baseline_amount = round(mean_30d, 2)
            normative_baseline["amount"] = baseline_amount
            normative_baseline["is_fraud"] = 0
            normative_baseline["scenario_tag"] = "ORGANIC_NORMAL"
            normative_baseline["haversine_velocity_kph"] = 0.0
            normative_baseline["ip_distance_from_home_km"] = 0.0
            normative_baseline["is_cross_border"] = False
            normative_baseline["avs_match_code"] = "Y"
            normative_baseline["billing_shipping_match"] = 1
            normative_baseline["cvv_match_flag"] = 1
            normative_baseline["flow_latent_u"] = [round(float(v), 5) for v in u_star]
            normative_baseline["flow_x_cf"] = [round(float(v), 5) for v in x_cf]

            if is_fraud == 1:
                counterfactual_mode = "ADVERSARIAL_INSERTION"
            else:
                counterfactual_mode = "ORGANIC_BASELINE"
            cf_input_deltas["amount"] = round(amount - baseline_amount, 2)
            cf_input_deltas["haversine_velocity_kph"] = round(effective_velocity, 2)
            cf_input_deltas["ip_distance_from_home_km"] = round(ip_dist - 0.0, 2)
            cf_input_deltas["billing_shipping_mismatch"] = float(bill_mismatch)
            cf_input_deltas["avs_mismatch_flag"] = float(avs_mismatch)
            cf_input_deltas["cvv_mismatch_flag"] = float(cvv_mismatch)

        # 4. Active Causal Drivers
        active_parents: List[str] = []
        best_driver = "baseline"
        max_contrib = 0.0

        for feat_name, phi_p in shapley_prob.items():
            if phi_p > 0.005:
                active_parents.append(feat_name)
            if phi_p > max_contrib:
                max_contrib = phi_p
                best_driver = feat_name

        if best_driver == "baseline":
            min_contrib = 0.0
            for feat_name, phi_p in shapley_prob.items():
                if phi_p < min_contrib:
                    min_contrib = phi_p
                    best_driver = feat_name

        currency_sym = "₹" if record.get("currency") == "INR" else "$"

        # 5. Narrative Explanation Generation
        narrative = self._generate_narrative(
            scenario_tag=scenario_tag,
            is_fraud=is_fraud,
            driver=best_driver,
            active_parents=active_parents,
            amount=amount,
            currency_sym=currency_sym,
        )

        return CausalGroundTruth(
            is_fraud=is_fraud,
            scenario_tag=scenario_tag,
            risk_score=round(risk_score, 6),
            base_risk=round(base_risk, 6),
            logit_z=round(logit_z, 4),
            base_logit=round(self.base_logit, 4),
            analytical_shapley_log_odds=shapley_logit,
            analytical_shapley_probability=shapley_prob,
            counterfactual_input_deltas=cf_input_deltas,
            dominant_causal_driver=best_driver,
            active_causal_parents=active_parents,
            counterfactual_mode=counterfactual_mode,
            counterfactual_twin=normative_baseline,
            normative_baseline=normative_baseline,
            explanation_narrative=narrative,
            latent_shapley_attributions=latent_shapley,
            pearlian_consistency_error=round(consistency_error, 12),
        )

    def _generate_narrative(
        self,
        scenario_tag: str,
        is_fraud: int,
        driver: str,
        active_parents: List[str],
        amount: float,
        currency_sym: str = "$",
    ) -> str:
        """Generates clear, analyst-readable causal narrative."""
        if is_fraud == 0:
            if "HARD_NEGATIVE" in scenario_tag:
                return (
                    f"Legitimate high-ticket or travel outlier ({currency_sym}{amount:.2f}). "
                    f"Elevated features ({', '.join(active_parents) if active_parents else driver}) "
                    f"are mitigated by authentic EMV cryptographic verification."
                )
            return "Routine cardholder transaction consistent with historical behavioral baseline."

        if scenario_tag == "CARD_TESTING_BURST":
            return (
                f"Automated card testing probe ({currency_sym}{amount:.2f}). Attacker probed card validity "
                f"at a low-friction merchant; primary causal driver: {driver}."
            )
        elif scenario_tag == "ACCOUNT_TAKEOVER":
            return (
                f"Account Takeover exploitation ({currency_sym}{amount:.2f}). Transaction initiated from an anomalous "
                f"offshore IP; primary causal driver: {driver}."
            )
        elif scenario_tag == "COUNTERFEIT_CLONE" or scenario_tag == "IMPOSSIBLE_TRAVEL":
            return (
                f"Counterfeit magstripe clone ({currency_sym}{amount:.2f}). Terminal swipe executed at remote POS "
                f"concurrent with local cardholder activity; primary causal driver: {driver}."
            )
        elif scenario_tag == "SLEEPER_BUST_OUT":
            return (
                f"Credit line bust-out ({currency_sym}{amount:.2f}). High-utilization balance drain; "
                f"primary causal driver: {driver}."
            )
        return f"Adversarial attack ({scenario_tag}, {currency_sym}{amount:.2f}) driven by {driver}."

    def score(self, record: Dict[str, Any]) -> float:
        """Computes continuous risk score using the heuristic polynomial."""
        result = self.evaluate(record)
        return result.risk_score

    def decide(self, record: Dict[str, Any], threshold: float = 0.50) -> str:
        """Returns ISO 8583 response code based on risk threshold."""
        score = self.score(record)
        if score >= threshold:
            return "59"  # Suspected Fraud
        return "00"  # Approved


# Backward compatibility aliases
StructuralCausalEngine = HeuristicBankScorer
CounterfactualCausalEngine = HeuristicBankScorer

