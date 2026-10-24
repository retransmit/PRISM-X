# Contributing to Prism Generator

Thank you for your interest in contributing to **Prism Generator**!

Prism Generator is an open-source research and engineering platform designed to simulate realistic payment card networks, adversarial dynamics, and causal ground-truth benchmarks for fraud detection.

---

## 1. The Core Mandate: Anti-Astronaut Grounding

All contributions must adhere strictly to the **Anti-Astronaut Grounding Mandate** codified in [`AGENTS.md`](AGENTS.md):

1. **Zero Theoretical Buzzwords:** Do not introduce speculative or non-implementable concepts unless backed by executable Python code.
2. **Physical & Financial Units:** Every parameter must have concrete physical or financial units (e.g., USD cents, INR paisa, seconds, km/h, probabilities in $[0.0, 1.0]$).
3. **Real-World Banking Plumbing:** Models must reflect actual payment rails: ISO 8583 response codes (`00` Approved, `05` Do Not Honor, `51` Insufficient Funds, `57` Transaction Not Permitted, `59` Suspected Fraud), AVS (Address Verification), CVV validation, 3DS 2.x exemption rules, and dual-message pre-auth holds vs. financial settlement.
4. **Specification-First Development:** All new scenarios or products must first be defined in `spec/` before writing implementation logic.
5. **Deterministic Verification:** Every behavioral claim must be proven via automated `pytest` assertions.

---

## 2. Development Setup

### Clone the Repository
```bash
cd generator                   # inside a checkout of this repository
```

### Install in Editable Mode
```bash
pip install -e ".[dev,docs]"
```
Add `benchmark` and `gui` when working on those areas (`.[dev,docs,benchmark,gui]`).

### Run the Test Suite
```bash
python -m pytest tests/ -v -ra
```
All tests must pass before submitting a pull request. For a faster inner loop that
skips the long end-to-end simulations:
```bash
python -m pytest tests/ -q -m "not slow"
```
The full suite is the gate; the fast selection is only a convenience. Read the count
and skip reasons pytest reports rather than trusting a number written here.

### External Corpora
`corpora/` is gitignored because the real datasets are hundreds of megabytes. Tests
that need the full ULB or IEEE-CIS files skip with an explicit reason when those files
are absent; the backbone replay invariants still run against the synthetic ARFF
fixture committed in the test suite. CI reports every skip (`-ra`) rather than hiding
them. To run the real-corpus tests locally, place the files where `spec/20` and the
test modules name them and rerun the suite.

### Run Operational Invariant Verification
```bash
python scripts/verify_grounded_invariants.py
```
All 37 grounded banking invariants must report `100% PASS RATE`.

### Run Forensic Realness Audit
```bash
python scripts/audit_fraud_realness.py
```

---

## 3. Submitting Pull Requests

1. **Fork the repository** and create a feature branch (`git checkout -b feature/your-feature-name`).
2. **Commit your changes** with descriptive commit messages.
3. **Ensure 100% test coverage** for any new features or bug fixes.
4. **Push to your fork** and submit a Pull Request against the `main` branch.

---

## 4. Code of Conduct

Prism Generator welcomes contributors from all backgrounds. We expect all participants to communicate with respect, clarity, and constructive feedback.
