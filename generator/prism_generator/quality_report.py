"""Distributional Quality, Internal Diversity, and Class Overlap Report.

Implements Dimension 5 verification protocols:
1. Internal Diversity: Mean pairwise Euclidean distance across normalized continuous feature vectors.
2. Categorical Shannon Entropy: Information richness across channel, MCC, and response codes.
3. Class Overlap (Bhattacharyya Coefficient): Verifies realistic statistical overlap between
   legitimate transactions and adversarial fraud (preventing hyper-separable simulation traps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class QualitySummary:
    """Summary metrics of synthetic data quality, diversity, and class overlap."""
    n_samples: int
    fraud_rate: float
    internal_diversity: float
    column_entropies: Dict[str, float]
    class_overlap_bhattacharyya: Dict[str, float]
    diversity_passed: bool
    entropy_passed: bool
    overlap_passed: bool
    overall_passed: bool

    def format_markdown(self) -> str:
        """Renders publication-grade GitHub Flavored Markdown report."""
        verdict = "**PASS (CERTIFIED HEALTHY)**" if self.overall_passed else "**FAIL (ANOMALY DETECTED)**"
        lines = [
            "# Prism Generator Distributional Quality & Realism Report",
            "",
            f"**Audit Status:** {verdict}  ",
            f"**Evaluated Samples:** `{self.n_samples:,}` | **Empirical Fraud Prevalence:** `{self.fraud_rate * 100:.2f}%`",
            "",
            "---",
            "",
            "## 1. Internal Diversity Suite",
            "",
            "| Metric | Empirical Value | Acceptance Bound | Description | Status |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **Mean Pairwise L2 Distance** | `{self.internal_diversity:.4f}` | >= 0.200 | Sampled pairwise diversity across normalized continuous attributes | {'PASS' if self.diversity_passed else 'FAIL'} |",
            "",
            "---",
            "",
            "## 2. Categorical Shannon Entropies (Bits)",
            "",
            "| Column Attribute | Empirical Entropy (Bits) | Bound | Description |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for col, ent in sorted(self.column_entropies.items()):
            lines.append(f"| `{col}` | `{ent:.4f}` | > 0.10 | Shannon information content $H(X)$ |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Realism Overlap Suite (Bhattacharyya Coefficient $BC \\in [0, 1]$)",
            "",
            "| Feature | Empirical Overlap $BC$ | Realism Bound | Interpretation | Status |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for feat, bc in sorted(self.class_overlap_bhattacharyya.items()):
            status = "PASS" if bc >= 0.15 else "FAIL"
            interp = "Realistic statistical fog" if bc < 0.95 else "Common baseline overlap"
            if bc < 0.15:
                interp = "Hyper-separable shortcut!"
            lines.append(f"| `{feat}` | `{bc:.4f}` | >= 0.150 | {interp} | {status} |")

        lines.extend([
            "",
            "---",
            "",
            f"### Final Quality Verdict: {verdict}",
        ])
        return "\n".join(lines)


class QualityReport:
    """Computes internal diversity, categorical entropies, and class overlap."""

    CONTINUOUS_COLUMNS = [
        "amount",
        "tx_count_1h",
        "tx_count_24h",
        "haversine_velocity_kph",
        "ip_distance_from_home_km",
        "vaai_score",
    ]

    CATEGORICAL_COLUMNS = [
        "channel_type",
        "mcc",
        "response_code",
        "asn_type",
    ]

    @classmethod
    def compute_internal_diversity(
        cls,
        records: List[Dict[str, Any]],
        n_pairs: int = 1000,
        seed: int = 42,
    ) -> float:
        """Calculates average L2 Euclidean distance between random pairs of normalized feature vectors."""
        if len(records) < 2:
            return 0.0

        matrix: List[List[float]] = []
        for r in records:
            row = [float(r.get(c, 0.0)) for c in cls.CONTINUOUS_COLUMNS]
            matrix.append(row)

        X = np.array(matrix, dtype=np.float64)
        c_min = np.min(X, axis=0)
        c_max = np.max(X, axis=0)
        c_range = c_max - c_min
        c_range[c_range == 0] = 1.0

        X_norm = (X - c_min) / c_range

        rng = np.random.default_rng(seed)
        n = len(X_norm)
        pairs_m = min(n_pairs, n * (n - 1) // 2)
        idx_a = rng.integers(0, n, size=pairs_m)
        idx_b = rng.integers(0, n, size=pairs_m)

        # Ensure a != b
        mask = idx_a == idx_b
        idx_b[mask] = (idx_b[mask] + 1) % n

        diff = X_norm[idx_a] - X_norm[idx_b]
        dists = np.linalg.norm(diff, axis=1)
        return float(np.mean(dists))

    @classmethod
    def compute_column_entropies(cls, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates discrete Shannon entropy (base 2) for categorical fields."""
        entropies: Dict[str, float] = {}
        n = len(records)
        if n == 0:
            return {c: 0.0 for c in cls.CATEGORICAL_COLUMNS}

        for col in cls.CATEGORICAL_COLUMNS:
            counts: Dict[str, int] = {}
            for r in records:
                val = str(r.get(col, "UNKNOWN"))
                counts[val] = counts.get(val, 0) + 1

            h = 0.0
            for count in counts.values():
                p = count / n
                if p > 0:
                    h -= p * math.log2(p)
            entropies[col] = float(h)

        return entropies

    @classmethod
    def compute_bhattacharyya_coefficient(
        cls,
        x_legit: np.ndarray,
        x_fraud: np.ndarray,
        n_bins: int = 40,
    ) -> float:
        """Computes empirical Bhattacharyya coefficient BC = sum(sqrt(p_i * q_i)) over shared histogram bins."""
        if len(x_legit) == 0 or len(x_fraud) == 0:
            return 0.0

        low = min(float(np.min(x_legit)), float(np.min(x_fraud)))
        high = max(float(np.max(x_legit)), float(np.max(x_fraud)))

        if low == high:
            return 1.0

        bins = np.linspace(low, high, n_bins + 1)
        p, _ = np.histogram(x_legit, bins=bins, density=False)
        q, _ = np.histogram(x_fraud, bins=bins, density=False)

        p_norm = p / max(1, np.sum(p))
        q_norm = q / max(1, np.sum(q))

        bc = float(np.sum(np.sqrt(p_norm * q_norm)))
        return float(np.clip(bc, 0.0, 1.0))

    @classmethod
    def compute_class_overlaps(cls, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Computes Bhattacharyya overlap for continuous features between legitimate and fraud."""
        legit = [r for r in records if int(float(r.get("is_fraud", 0))) == 0]
        fraud = [r for r in records if int(float(r.get("is_fraud", 0))) == 1]

        overlaps: Dict[str, float] = {}
        for col in ["vaai_score", "amount", "haversine_velocity_kph", "ip_distance_from_home_km"]:
            v_legit = np.array([float(r.get(col, 0.0)) for r in legit], dtype=float)
            v_fraud = np.array([float(r.get(col, 0.0)) for r in fraud], dtype=float)
            overlaps[col] = cls.compute_bhattacharyya_coefficient(v_legit, v_fraud)

        return overlaps

    @classmethod
    def generate_quality_summary(
        cls,
        records: List[Dict[str, Any]],
        n_pairs: int = 1000,
        seed: int = 42,
    ) -> QualitySummary:
        """Generates comprehensive distributional quality and realism summary."""
        n_samples = len(records)
        n_fraud = sum(1 for r in records if int(float(r.get("is_fraud", 0))) == 1)
        fraud_rate = n_fraud / max(1, n_samples)

        diversity = cls.compute_internal_diversity(records, n_pairs=n_pairs, seed=seed)
        entropies = cls.compute_column_entropies(records)
        overlaps = cls.compute_class_overlaps(records)

        diversity_pass = bool(diversity >= 0.20)
        entropy_pass = bool(entropies.get("channel_type", 0.0) >= 0.80 and entropies.get("mcc", 0.0) >= 1.50)
        vaai_bc = overlaps.get("vaai_score", 0.0)
        amt_bc = overlaps.get("amount", 0.0)
        overlap_pass = bool(vaai_bc >= 0.15 and amt_bc >= 0.15)

        overall = bool(diversity_pass and entropy_pass and overlap_pass)

        return QualitySummary(
            n_samples=n_samples,
            fraud_rate=fraud_rate,
            internal_diversity=diversity,
            column_entropies=entropies,
            class_overlap_bhattacharyya=overlaps,
            diversity_passed=diversity_pass,
            entropy_passed=entropy_pass,
            overlap_passed=overlap_pass,
            overall_passed=overall,
        )
