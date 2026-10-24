"""Invertible Deep Structural Causal Model (DSCM) & Normalizing Flows for Prism Generator (EXPERIMENTAL).

WARNING: This module is experimental.
The Conditional RealNVP flow is an algebraic diffeomorphism that demonstrates mathematical
invertibility by construction, but is NOT an empirical density estimator fitted to real data.
For production causal counterfactuals, use direct simulation resimulation with surgical interventions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AffineCouplingLayer:
    """Vectorized conditioned affine coupling layer with exact analytical inverse."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        mask: np.ndarray,
        hidden_dim: int = 32,
        scale_bound: float = 2.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.dim = dim
        self.context_dim = context_dim
        self.mask = mask.astype(bool)
        self.scale_bound = scale_bound
        rng = rng or np.random.default_rng(42)

        # Transformation network input: masked x (dim) + context (context_dim)
        in_dim = dim + context_dim
        out_dim = dim

        # Stable Xavier / He weight initialization
        scale_w1 = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.W1 = rng.normal(0.0, scale_w1, size=(in_dim, hidden_dim)).astype(np.float64)
        self.W1_x = self.W1[:dim, :].copy()
        self.W1_c = self.W1[dim:, :].copy()
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)

        scale_w2 = math.sqrt(2.0 / (hidden_dim + out_dim))
        self.W2_s = rng.normal(0.0, scale_w2, size=(hidden_dim, out_dim)).astype(np.float64)
        self.b2_s = np.zeros(out_dim, dtype=np.float64)

        self.W2_t = rng.normal(0.0, scale_w2, size=(hidden_dim, out_dim)).astype(np.float64)
        self.b2_t = np.zeros(out_dim, dtype=np.float64)

        # Float mask: multiplying by 0/1 reproduces np.where(mask, ...) for the finite
        # values the codec emits, and saves an allocation per coupling evaluation. The
        # flow is evaluated per transaction, so this is not a micro-optimisation.
        self._mask_f = self.mask.astype(np.float64)

    def _context_term(self, context: np.ndarray) -> np.ndarray:
        """Context contribution (b1 + C W1_c), hoisted out of the per-sample work.

        A caller that reuses one context across many rows -- path integration over the
        quadrature nodes does exactly this -- computes it once instead of once per row.
        """
        return np.dot(context, self.W1_c) + self.b1

    def _compute_st(self, x_masked: np.ndarray, context_term: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes scale (s) and translation (t) from pre-masked x and a precomputed context term.

        Accepts a single row (dim,) or a stack of rows (B, dim); the mask and the bias
        broadcast over the leading dimension.
        """
        z1 = np.dot(x_masked, self.W1_x) + context_term
        h1 = np.where(z1 > 0.0, z1, z1 * 0.1)

        raw_s = np.dot(h1, self.W2_s) + self.b2_s
        s = self.scale_bound * np.tanh(raw_s)

        t = np.dot(h1, self.W2_t) + self.b2_t

        s_masked = np.where(self.mask, 0.0, s)
        t_masked = np.where(self.mask, 0.0, t)

        return s_masked, t_masked

    def forward(self, x: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward mapping y = F(x; C) and log-determinant."""
        s, t = self._compute_st(x * self._mask_f, self._context_term(context))
        # s and t are zero on the identity positions, so x*exp(0) + 0 == x there and the
        # explicit branch on the mask is unnecessary.
        y = x * np.exp(s) + t
        log_det = float(np.sum(s))
        return y, log_det

    def inverse(self, y: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, float]:
        """Exact closed-form inverse mapping x = F^{-1}(y; C) and log-determinant."""
        s, t = self._compute_st(y * self._mask_f, self._context_term(context))
        x = (y - t) * np.exp(-s)
        log_det = -float(np.sum(s))
        return x, log_det

    def forward_batch(self, x: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The `forward` mapping evaluated for a stack of rows; log-determinants per row."""
        s, t = self._compute_st(x * self._mask_f, self._context_term(context))
        return x * np.exp(s) + t, np.sum(s, axis=1)

    def inverse_batch(self, y: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The `inverse` mapping evaluated for a stack of rows; log-determinants per row."""
        s, t = self._compute_st(y * self._mask_f, self._context_term(context))
        return (y - t) * np.exp(-s), -np.sum(s, axis=1)


class ConditionalRealNVPFlow:
    """Experimental Invertible Architecture with exact algebraic invertibility."""

    def __init__(
        self,
        dim: int = 8,
        context_dim: int = 8,
        num_layers: int = 6,
        hidden_dim: int = 32,
        seed: int = 42,
    ):
        self.dim = dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.layers: List[AffineCouplingLayer] = []
        for i in range(num_layers):
            mask = np.zeros(dim, dtype=bool)
            if i % 2 == 0:
                mask[::2] = True
            else:
                mask[1::2] = True

            layer = AffineCouplingLayer(
                dim=dim,
                context_dim=context_dim,
                mask=mask,
                hidden_dim=hidden_dim,
                scale_bound=1.75,
                rng=self.rng,
            )
            self.layers.append(layer)

    def forward(self, x: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, float]:
        curr = x.copy().astype(np.float64)
        total_log_det = 0.0
        for layer in self.layers:
            curr, log_det = layer.forward(curr, context)
            total_log_det += log_det
        return curr, total_log_det

    def inverse(self, u: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, float]:
        curr = u.copy().astype(np.float64)
        total_log_det = 0.0
        for layer in reversed(self.layers):
            curr, log_det = layer.inverse(curr, context)
            total_log_det += log_det
        return curr, total_log_det

    def forward_batch(self, x: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward mapping for a stack of rows, with a per-row log-determinant.

        Used by path integration, which walks the same flow at many quadrature nodes for
        one observation. Each coupling layer is affine in the row dimension, so a batch
        of rows carries out the same arithmetic a single row would, node by node.
        """
        curr = x.copy().astype(np.float64)
        total_log_det = np.zeros(len(curr), dtype=np.float64)
        for layer in self.layers:
            curr, log_det = layer.forward_batch(curr, context)
            total_log_det = total_log_det + log_det
        return curr, total_log_det

    def inverse_batch(self, u: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse mapping for a stack of rows, with a per-row log-determinant."""
        curr = u.copy().astype(np.float64)
        total_log_det = np.zeros(len(curr), dtype=np.float64)
        for layer in reversed(self.layers):
            curr, log_det = layer.inverse_batch(curr, context)
            total_log_det = total_log_det + log_det
        return curr, total_log_det

    def log_prob(self, x: np.ndarray, context: np.ndarray) -> float:
        u, log_det = self.forward(x, context)
        dim = len(x)
        log_prior = -0.5 * (dim * math.log(2.0 * math.pi) + float(np.sum(u ** 2)))
        return log_prior + log_det

    def abduce(self, x_obs: np.ndarray, context_obs: np.ndarray) -> np.ndarray:
        u_star, _ = self.forward(x_obs, context_obs)
        return u_star

    def predict_counterfactual(
        self,
        u_star: np.ndarray,
        context_counterfactual: np.ndarray,
    ) -> np.ndarray:
        x_cf, _ = self.inverse(u_star, context_counterfactual)
        return x_cf

    def derive_counterfactual_twin(
        self,
        x_obs: np.ndarray,
        context_obs: np.ndarray,
        context_counterfactual: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_star = self.abduce(x_obs, context_obs)
        x_cf = self.predict_counterfactual(u_star, context_counterfactual)
        deltas = x_obs - x_cf
        return x_cf, u_star, deltas

    def verify_pearlian_consistency(self, x: np.ndarray, context: np.ndarray) -> float:
        u, _ = self.forward(x, context)
        x_rec, _ = self.inverse(u, context)
        return float(np.max(np.abs(x - x_rec)))


class TransactionFlowFeatureCodec:
    """Translates between tabular transaction dictionaries and continuous Flow representations."""

    FEATURE_NAMES: List[str] = [
        "log_amount",
        "velocity_kph",
        "log_ip_distance",
        "tx_count_1h",
        "sum_24h_ratio",
        "hour_sin",
        "hour_cos",
        "friction_score",
    ]

    CONTEXT_NAMES: List[str] = [
        "is_fraud",
        "is_cp",
        "is_cross_border",
        "is_high_risk_mcc",
        "is_night",
        "emv_crypto_verified",
        "card_tier_affluent",
        "region_in",
    ]

    @classmethod
    def encode(cls, record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        amount = float(record.get("amount", 25.0))
        log_amount = math.log(max(1.0, amount))

        velocity = float(record.get("haversine_velocity_kph", 0.0))
        velocity_clamped = min(900.0, max(0.0, velocity))

        ip_dist = float(record.get("ip_distance_from_home_km", 0.0))
        log_ip = math.log(max(1.0, ip_dist))

        count_1h = float(record.get("tx_count_1h", 0.0))

        credit_limit = float(record.get("credit_limit", 5000.0))
        sum_24h = float(record.get("tx_amount_sum_24h", 0.0))
        sum_ratio = sum_24h / max(credit_limit, 500.0)

        hour = float(record.get("hour_of_day", 14.0))
        angle = 2.0 * math.pi * hour / 24.0
        hour_sin = math.sin(angle)
        hour_cos = math.cos(angle)

        avs = 1.0 if record.get("avs_match_code") in ("N", "U") else 0.0
        cvv = 1.0 if int(record.get("cvv_match_flag", 1)) == 0 else 0.0
        friction = avs * 1.5 + cvv * 2.0

        x = np.array([
            log_amount,
            velocity_clamped,
            log_ip,
            count_1h,
            sum_ratio,
            hour_sin,
            hour_cos,
            friction,
        ], dtype=np.float64)

        is_fraud = float(record.get("is_fraud", 0))
        ch = str(record.get("channel_type", ""))
        is_cp = 1.0 if ch.startswith("CP") else 0.0
        is_cross = 1.0 if bool(record.get("is_cross_border", False)) else 0.0

        mcc = int(record.get("mcc", 5411))
        is_high_risk = 1.0 if mcc in (5094, 6051, 5732, 7995, 8398) else 0.0
        is_night = 1.0 if (1 <= int(hour) <= 5) else 0.0

        arqc = 1.0 if (is_cp and is_fraud == 0 and ch != "CP_POS_MAGSTRIPE") else 0.0
        cohort = str(record.get("cohort_id", ""))
        is_affluent = 1.0 if ("AFFLUENT" in cohort or "TECH" in cohort) else 0.0
        region = str(record.get("currency", "USD"))
        is_in = 1.0 if region == "INR" else 0.0

        C = np.array([
            is_fraud,
            is_cp,
            is_cross,
            is_high_risk,
            is_night,
            arqc,
            is_affluent,
            is_in,
        ], dtype=np.float64)

        return x, C

    @classmethod
    def decode_counterfactual(
        cls,
        x_cf: np.ndarray,
        factual_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        cf = dict(factual_record)
        cf["is_fraud"] = 0
        cf["scenario_tag"] = "ORGANIC_NORMAL"

        amount_recovered = math.exp(float(x_cf[0]))
        cf["amount"] = round(max(0.50, amount_recovered), 2)
        cf["haversine_velocity_kph"] = round(max(0.0, float(x_cf[1])), 2)

        ip_dist_recovered = math.exp(float(x_cf[2]))
        cf["ip_distance_from_home_km"] = round(max(0.0, ip_dist_recovered), 2)

        cf["tx_count_1h"] = int(max(0, round(float(x_cf[3]))))
        cf["avs_match_code"] = "Y"
        cf["cvv_match_flag"] = 1
        cf["billing_shipping_match"] = 1
        cf["auth_response_code"] = "00"
        cf["response_code"] = "00"

        return cf


class LatentAumannShapleyAttributor:
    """Computes path-integrated Shapley values along the latent flow geodesic (Experimental)."""

    def __init__(self, flow: ConditionalRealNVPFlow, n_steps: int = 64):
        self.flow = flow
        self.n_steps = n_steps
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        self._t_nodes = 0.5 * (nodes + 1.0)
        self._weights = 0.5 * weights

    def attribute(
        self,
        x_obs: np.ndarray,
        context: np.ndarray,
        scorer_fn: Any,
        u_base: Optional[np.ndarray] = None,
        u_obs: Optional[np.ndarray] = None,
        scorer_batch_fn: Any = None,
    ) -> Dict[str, float]:
        """Aumann-Shapley attribution from `u_base` to `u_obs`.

        `u_obs` is accepted because the caller usually already holds it (abduction
        computes exactly this forward map), and `scorer_batch_fn`, when given, scores a
        stack of rows in one call and must agree with `scorer_fn` row for row. Together
        they turn n_steps x (1 + dim) interpreter-level scorer calls per record into two
        array calls, which is what made attribution affordable during batch generation.
        """
        dim = len(x_obs)
        if u_obs is None:
            u_obs, _ = self.flow.forward(x_obs, context)
        if u_base is None:
            u_base = np.zeros(dim, dtype=np.float64)

        x_base, _ = self.flow.inverse(u_base, context)
        score_base = scorer_fn(x_base)
        score_obs = scorer_fn(x_obs)
        total_delta = score_obs - score_base

        # One batched inverse for all quadrature nodes: each layer is affine in the row
        # dimension, so 64 single-row inverses and one 64-row inverse differ only in the
        # order of the same floating-point sums. This loop used to cost about seven
        # milliseconds per attributed record, which dominated batch generation.
        u_nodes = (
            (1.0 - self._t_nodes)[:, None] * u_base[None, :]
            + self._t_nodes[:, None] * u_obs[None, :]
        )
        x_nodes, _ = self.flow.inverse_batch(u_nodes, context)

        phi = np.zeros(dim, dtype=np.float64)
        eps = 1e-4

        if scorer_batch_fn is not None:
            # All centres and all n_steps x dim perturbed rows in two scored stacks:
            # (n, dim) and (n * dim, dim), instead of n * (1 + dim) calls.
            centres = np.asarray(scorer_batch_fn(x_nodes), dtype=np.float64)
            perturbed = np.repeat(x_nodes, dim, axis=0)
            row_idx = np.arange(len(perturbed))
            col_idx = np.tile(np.arange(dim), len(x_nodes))
            perturbed[row_idx, col_idx] += eps
            values = np.asarray(scorer_batch_fn(perturbed), dtype=np.float64).reshape(len(x_nodes), dim)

            grads = (values - centres[:, None]) / eps
            # Same sum as the loop below: sum_n (x_obs - x_base) * grad_n * w_n
            phi = (x_obs - x_base) * (grads.T @ self._weights)
        else:
            for x_t, w in zip(x_nodes, self._weights):
                grad_x = np.zeros(dim, dtype=np.float64)
                s_center = scorer_fn(x_t)
                for d in range(dim):
                    x_pert = x_t.copy()
                    x_pert[d] += eps
                    s_pert = scorer_fn(x_pert)
                    grad_x[d] = (s_pert - s_center) / eps

                phi += (x_obs - x_base) * grad_x * w

        phi_sum = float(np.sum(phi))
        if abs(phi_sum) > 1e-9 and abs(total_delta) > 1e-9:
            phi = phi * (total_delta / phi_sum)

        return {name: float(phi[i]) for i, name in enumerate(TransactionFlowFeatureCodec.FEATURE_NAMES)}
