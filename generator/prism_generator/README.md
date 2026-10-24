# Prism Generator: Discrete-Event Payment Fraud Simulation & Risk Attribution Benchmark

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](../pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-140-brightgreen.svg)]()

**Prism Generator** is a high-performance, discrete-event payment simulation framework that generates realistic card transaction feeds, banking authorization protocol fields, and ground-truth feature risk attributions for evaluating machine learning models.

---

## 🌟 Key Architecture & Capabilities

1. **Strictly Monotonic Discrete-Event Queue (`heapq`)**:
   Operates on a 64-bit integer microsecond priority queue with stable sequence tie-breaking. Guarantees global temporal monotonicity ($t_0 \le t_1 \le \dots \le t_N$) and eliminates out-of-order time-travel artifacts.

2. **Physical Transit Velocity Limits ($t_{\text{avail}}[i]$)**:
   Enforces physical lock-ahead dwell and transit times per cardholder ($\Delta t \ge 120\,\text{s}$ on physical Card-Present swipes), guaranteeing that legitimate in-person transactions never exceed commercial travel velocity ($\le 900\,\text{km/h}$).

3. **Closed-Loop Behavioral Feedback**:
   - **Cardholder Profiles**: Model 24-hour diurnal schedules, multi-stop shopping trip clustering, travel/vacation states, and multi-modal fraud discovery latencies (SMS push vs. mobile app check vs. billing statement).
   - **Adversarial Fraud Playbooks**: Implements 10 cybercrime playbooks (Card Testing Probes, Account Takeover with silent baking, Sleeper Bust-Outs, Apple Pay Yellow Path, etc.). Attackers adapt to bank response codes (bisection amount decay on ISO 51, gateway hopping on 3DS challenges).
   - **Bank Decision Engine**: Multi-tier evaluation enforcing format checks, CVV verification, transit velocity limits, credit limits, and real-time ML risk scoring with EMV 3DS 2.x risk challenges.

4. **Authentic Hard Negatives**:
   Simulates legitimate high-spend and high-velocity outliers (e.g., cross-border vacation travel, emergency home repairs) that exhibit anomalous statistical signatures but are validated by genuine EMV chip cryptograms.

5. **Ground-Truth Feature Risk Attributions & Baseline Deltas**:
   For every simulated fraud transaction, the engine computes the exact difference relative to the cardholder's uncompromised 30-day baseline profile:
   $$\Delta \mathbf{x} = \mathbf{x}_{\text{fraud}} - \mathbf{x}_{\text{baseline}}$$
   Provides a built-in benchmark evaluator (`GroundTruthXAIEvaluator`) computing:
   - **Support Recovery**: Precision@k, Recall@k, F1@k
   - **Ranking Fidelity**: Kendall's $\tau_b$ and Spearman's $\rho$
   - **Vector Distances**: Directional Cosine Similarity, normalized $L_1/L_2$, and Relative Attribution Error (RAE)

---

## 🚀 Quick Start

### Installation
```bash
cd Prism Generator
pip install -e .
```

### Batch Synthesis (Python API)
```python
from prism_generator import DiscreteEventEngine

engine = DiscreteEventEngine(n_cards=1000, n_merchants=150, seed=42)
records = engine.generate_batch(
    n_transactions=5000,
    fraud_prevalence=0.03,
    time_span_days=30,
)

print(f"Generated {len(records)} strictly monotonic transactions.")
print("Sample record:", records[0])
```

### Benchmarking an XAI Model Against Ground Truth
```python
import numpy as np
from prism_generator import GroundTruthXAIEvaluator

evaluator = GroundTruthXAIEvaluator()

# Ground-truth causal attribution vector from simulation
phi_star = np.array([0.45, 0.35, 0.15, 0.00, 0.05])

# Post-hoc model attribution (e.g. from TreeSHAP)
phi_model = np.array([0.40, 0.30, 0.10, 0.02, 0.08])

result = evaluator.evaluate_instance(phi_model, phi_star, k_values=(2, 3))
print("Precision@2:", result.precision_at_k[2])
print("Cosine Similarity:", result.cosine_similarity)
print("Kendall Tau:", result.kendall_tau)
```

---

## 🧪 Verification & Test Suite

Run the synthesizer suite -- 140 tests here, 299 in the repository as a whole:
```bash
python -m pytest tests/test_synthesizer/ -v
```
