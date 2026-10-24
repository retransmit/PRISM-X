# Prism Generator Workspace Rules: Grounded Adversarial & Behavioral Simulation

## 1. The Core Mandate: Anti-Astronaut Grounding
- **Zero Sci-Fi / Theoretical Buzzwords:** Do not introduce speculative or non-implementable concepts (e.g., Quantum Amplitude Estimation, edge neuromorphic Loihi chips, sheaf cohomology) unless they are actively being written as executable Python code.
- **The Implementability Veto:** Any research concept must be representable as an Agent State Machine, a parameterized probability distribution, an event queue handler, or an explicit invariant assertion in Python/NumPy/Polars. If it cannot be executed or tested locally, it is forbidden.
- **Physical & Financial Units:** Every parameter must have concrete physical or financial units (e.g., USD/EUR cents, seconds, km/h, probabilities in [0.0, 1.0]). No dimensionless or ungrounded magic numbers.
- **Real-World Banking Plumbing:** Models must reflect actual payment rails: ISO 8583 response codes (e.g., 00 Approved, 05 Do Not Honor, 51 Insufficient Funds, 59 Suspected Fraud), AVS (Address Verification), CVV validation, 3DS 2.x exemption rules, and dual-message pre-auth holds vs. financial settlement.

---

## 2. Specification-First Development Workflow
All work proceeds in strictly scoped **Vertical Slices** using the following lifecycle:

1. **Spec in `spec/`:** Define the parameters, state machines, and real-world justifications in typed YAML/Pydantic schemas before writing any logic.
2. **Deterministic Tests:** Write automated pytest assertions proving the behavior (e.g., a cardholder in state X encountering merchant Y produces exact sequence Z).
3. **Minimal Implementation:** Implement the logic in the simulator engine without bloated abstractions.
4. **Data Inspection:** Generate a sample batch of transactions and inspect the raw output. Verify that distributions, edge cases, and hard negatives reflect reality.

---

## 3. Persistent Knowledge Base & Tracking
- The single source of truth is located in the [`spec/`](spec) directory.
- [`spec/active_slice.yaml`](spec/active_slice.yaml) tracks the currently active vertical slice, its acceptance criteria, and its status.
- Before proposing major changes, read `spec/active_slice.yaml` to maintain continuity without re-reading the entire repository.

---

## 4. Zero Arbitrary Assumptions & Rigorous Self-Interrogation
- **No Arbitrary Counts or Pre-Baked Categories:** Never invent arbitrary numbers (e.g., "top 5 playbooks", "6 card tiers", "6 personas") for presentation convenience. Let the empirical domain data, payment network specifications, and regulatory standards define the true scope.
- **Interrogate Every Design Choice:** Before proposing any number, threshold, or taxonomy, ask: *Where did this come from? Did I generate this because it looks neat, or is it grounded in an official payment manual, economic survey, or observed adversarial practice?* If an assumption is unverified, flag it explicitly as a hypothesis to be researched, never as an established fact.

