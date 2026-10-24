# Pillar 1 scored against real payment data

**Status:** measurement, not a certification. **Date:** 2026-10-10. **Region graded against:** US (spec/09 reference).

## Why this was run

Pillar 1 of the certification grader (`spec/18_benchmark_reporting.yaml`) compares our
synthetic stream against reference distributions taken from payment-rail sources. The
benchmark currently reports `NON_CERTIFIED_FAIL`, with pillar 1 among the failing
pillars. A failing grade has two possible causes the report cannot tell apart:

1. our generator is off, or
2. the reference does not describe real card traffic either.

`scripts/score_external_corpus.py` runs the **grader's own function**
(`UnifiedBenchmarkRunner._evaluate_data_fidelity`) on other corpora, so every row in the
table below is produced by identical code against thresholds read from `spec/18`. The
only thing that differs between columns is the data.

## Command

```bash
python scripts/score_external_corpus.py \
  --reference-control 1500 \
  --synthetic 1500 --synthetic 50000 \
  --ulb /tmp/opencode/corpora/creditcard.arff \
  --ulb-legit /tmp/opencode/corpora/creditcard.arff \
  --csv /tmp/opencode/corpora/banksim.csv --amount-col amount --time-col step \
       --time-unit hours --label-col fraud \
  --json /tmp/opencode/pillar1_scorecard.json
```

## Corpora

| column | what it is | source | sha256 |
|---|---|---|---|
| Prism Generator synthetic | this repo's engine, seed 42, `fraud_prevalence=0.07` | `prism_generator.engine.SimulationEngine` | n/a (generated) |
| ULB real | 284,807 real European card transactions, 2 days, 0.1727% flagged fraud | OpenML dataset 1597, file 1673544 (`https://openml.org/data/v1/download/1673544/creditcard.arff`) | `fdaf12730dc1fc426f318b71349f24f5c5fd00aa1152940be7e7509ae3d89d2a` |
| BankSim | 594,643 rows from a competing agent-based simulator (López-Rojas et al.), calibrated on a Spanish bank's aggregates | `https://raw.githubusercontent.com/atavci/fraud-detection-on-banksim-data/master/Data/synthetic-data-from-a-financial-payment-system/bs140513_032310.csv` | `e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377` |
| reference-sampler control | **not a generator**: amounts and inter-arrivals drawn from the reference itself | this script | n/a |

Both downloads were unauthenticated; no account was used.

## Results

| metric | limit | control n=1,500 | ours n=1,500 | ours n=50,000 | ULB real (all) | ULB real (legit) | BankSim |
|---|---|---|---|---|---|---|---|
| log10(1+amount) Wasserstein | ≤ 0.150 | 0.0000 **PASS** | 0.1508 FAIL | 0.1525 FAIL | **0.3745 FAIL** | 0.3740 FAIL | 0.2649 FAIL |
| log10(1+Δt seconds) Wasserstein | ≤ 0.150 | 0.0016 **PASS** | 0.2295 FAIL | 0.1161 **PASS** | 0.1716 FAIL | 0.1715 FAIL | 3.2941 FAIL |
| MCC Jensen-Shannon | ≤ 0.050 | n/a | 0.0747 FAIL | 0.0869 FAIL | n/a | n/a | n/a |
| channel Jensen-Shannon | ≤ 0.050 | n/a | 0.0131 **PASS** | 0.0707 FAIL | n/a | n/a | n/a |
| correlation Frobenius error | ≤ 1.250 | n/a | 0.0808 **PASS** | 0.0892 **PASS** | n/a | n/a | n/a |
| **verdict** (comparable metrics) | | **PASS** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

Corpus detail as printed by the run:

| corpus | n | span | median amount | dt_min | dt_med |
|---|---|---|---|---|---|
| ours n=1,500 | 1,500 | 10.39 d | 30.33 USD | 0.3 s | 228.53 s |
| ours n=50,000 | 50,000 | 51.75 d | 37.82 USD | 0.01 s | 45.91 s |
| ULB real (all) | 284,807 | 2.00 d | 22.00 EUR | 1 s | 1 s |
| BankSim | 594,643 | 7.46 d | 26.90 (currency unlabelled) | **3600 s** | **3600 s** |
| reference-sampler control | 1,500 | 12.00 d | 42.65 USD | 0.13 s | 496.56 s |

`n/a` = the corpus does not carry that metric's inputs, so the metric is excluded from
the verdict instead of being scored on the grader's per-row defaults (`mcc` → 5411,
`pos_entry_mode` → "01", correlation columns → 0.0, each of which makes its own metric
trivially 0).

## What the numbers say

1. **Real data fails the amount gate, and fails it worse than we do.** ULB scores
   0.3745 against a 0.150 limit (2.5× our 0.1508). Dropping ULB's fraud rows changes
   nothing (0.3740), so it is not fraud enrichment. The gate measures distance to the
   hardcoded `lognormal(mu=3.75, sigma=0.85)` USD reference, and on this corpus real
   card amounts are not that distribution: ULB's median is 22 against the reference's
   42.65, with a heavier right tail.

2. **The gate is gameable by construction.** The control draws straight from the
   reference and scores 0.0000 / 0.0016 → PASS, while being nothing like card traffic.
   Distance to the reference is therefore not evidence of realism.

3. **The inter-arrival gate conflates arrival shape with clock resolution.** BankSim's
   3.2941 is dominated by its hourly `step` (dt_min = dt_med = 3600 s), not by its
   arrival process; ULB's timestamps are whole seconds and its median Δt is 1 s, so the
   grader's `Δt > 0` filter drops every same-second pair and biases its score upward;
   our own Δt floor is 0.01 to 0.3 s. Three corpora, three clocks: the number is not
   comparable across them without the dt_min column.

4. **Sample size moves the verdicts at the grader's default n=1,500.** Our
   inter-arrival score goes 0.2295 (FAIL) → 0.1161 (PASS) at n=50,000, and channel JSD
   goes 0.0131 (PASS) → 0.0707 (FAIL): the JSD gates compare against a *uniform*
   distribution and are biased downward for small samples, so small runs are graded
   easier. The amount score is stable (0.1508 → 0.1525).

5. **Two gates cannot be run on any corpus but our own.** MCC and channel JSD are
   distances to uniform over the categories we emit; neither ULB nor BankSim carries
   MCC or POS-entry-mode codes, and the Frobenius gate needs our own five feature
   columns (`tx_count_1h`, `haversine_velocity_kph`, ...). Those three gates are
   self-referential to our schema: they can fail us, but no external data can
   validate them.

## What this does and does not establish

**Does:** pillar 1's continuous gates do not rank data by resemblance to real
transactions. On the only real corpus obtainable without an account, real transactions
fail the amount gate harder than our synthetic output, and a sampler of the reference
passes trivially. Closing pillar 1 by tuning the generator to the reference would move
the generator *away* from this real corpus.

**Does not:** that the reference itself is wrong
fraud-flagged, EUR-denominated extract and BankSim is calibrated on a Spanish bank;
neither is the Federal Reserve's US card population, so "real data fails" is
directional, not conclusive. It also says nothing about pillars 2 to 4: the pillar 4
relative-attribution-error failure is an internal-consistency result and is untouched
by this measurement.

**Not measured here:** pillar 1's remaining gates on external data (impossible for the
reasons in finding 5), and any corpus behind Kaggle authentication (IEEE-CIS would be
the natural second real corpus and needs an API key).

## Proposed next step (a decision for the slice owner, not taken here)

Re-derive pillar 1 against a real corpus instead of a hardcoded lognormal: grade
generators on distance-to-real (with the same clock and sample size on both sides),
keep a size-corrected statistic, and record the reference corpus and its hash in
`spec/18` the way thresholds are already recorded there. Until then, pillar 1's FAIL
should be read as "differs from the Fed lognormal", not "less realistic than real
data".

---

## Update 2026-10-15: the proposed next step was taken

The owner approved the re-derivation, with the additional constraint that the
reference should not rest on ULB's age. The corpus retrieval was therefore
re-attempted with credentials: `kagglehub`'s anonymous path still fails for
competition data (re-measured: `UnauthenticatedError`), but the Kaggle competition
API with the project owner's joined account fetches IEEE-CIS directly. Pillar 1's
reference is now the **IEEE-CIS / Vesta charity of real card-not-present traffic**:

| | old reference (until 2026-10-14) | active reference (2026-10-15) |
|---|---|---|
| corpus | hardcoded Fed lognormal(3.75, 0.85) USD | IEEE-CIS/Vesta `train_transaction.csv`, sha256 `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` |
| what it is | a parametric draw | 590,540 real CNP transactions, 182 days, 3.4990% fraud, median dt 14 s, USD |
| limit | static 0.150 | size-corrected: 99th percentile of bootstrap REAL-to-real W1 at the graded n, from contiguous corpus windows |
| real data passes? | never measured, and when measured: FAILED (ULB 0.3745) | by construction: contiguous real windows pass 8/9 (amount) and 9/9 (arrival) at n=2,000 |

Method notes fixed along the way, each found by measurement, not by preference:

- **Random-row subsets cannot grade arrivals.** An iid 2,000-row slice of
  IEEE-CIS has an inter-arrival shape dominated by the sampling gap, not by the
  stream (it failed its own gate at 0.0365 > 0.0207). The calibration therefore
  draws both graded and reference windows as *contiguous* ordered slices.
- **iid pairs alone underestimate a real stream's own spread.** With iid-only
  calibration, one of three real n=2,000 windows failed its amount bar; the
  amount limit was switched to contiguous-window calibration for the same
  reason as the arrival limit. The one marginal real-window fail (0.0925 vs
  0.0916) is a genuinely heterogeneous segment and is recorded, not tuned away.
- **The statistic is rate-invariant.** Both sides are normalised by their own
  median delta before the log transform, so a 0.038 tx/s real window and a
  0.002 tx/s protocol stream are compared on arrival shape, not rate.

### Measured 2026-10-15 (limit column is the calibrated value at each graded size)

| corpus | amount W1 | limit | arrival W1 | limit | verdict |
|---|---|---|---|---|---|
| Prism Generator synthetic, n=2,000 (protocol seed 42) | 0.3732 | 0.0916 FAIL | 0.0457 | 0.0446 FLIP | FAIL |
| reference-sampler control, n=2,000 | 0.2810 | 0.0916 FAIL | 0.0267 | 0.0446 PASS | FAIL |
| IEEE-CIS real window #0 (n=2,000) | 0.0111 | 0.0916 PASS | 0.0159 | 0.0412 PASS | PASS |
| IEEE-CIS real window #3 (n=2,000) | 0.0509 | 0.0916 PASS | 0.0305 | 0.0433 PASS | PASS |
| IEEE-CIS real window #6 (n=2,000) | 0.0925 | 0.0916 FAIL | 0.0403 | 0.0467 PASS | FAIL |
| IEEE-CIS real window (n=50,000) | 0.0090 | 0.0342 PASS | 0.0061 | 0.0165 PASS | PASS |
| ULB legit (2013, EUR, n=284,000) ,  comparative only | 0.5353 | 0.0069 FAIL | 0.1472 | 0.0159 FAIL | FAIL |

The inversion the 2026-10-10 table wanted is now the gate's actual behaviour:
real 2019 traffic passes its own bars, the sampler of the fictional reference
fails the amount bar, and the generator is graded on its true distance to real
card traffic. The 2019 reference also answers the owner's objection to ULB: no
currency conversion (USD native), 182 days rather than 2, 1-second timestamps
with a 14 s median delta instead of an all-second grid, and 2019 rather than
2013 data. ULB stays recorded in spec/18 for reproducing the tables above, not
for gating.

### Consequences for the certification grade

The grade at the pinned protocol remains `NON_CERTIFIED_FAIL` (as it was before,
measured `prism_generator benchmark -n 2000 --unified --seed 42`): the two continuous
failures are now *realism* failures against real traffic, with the violation
line naming the calibrated limit it was graded against:

- log-amount W1 0.3732 exceeds the 0.0916 maximum (5/5 seeds, solid)
- inter-arrival W1 0.0457 vs 0.0446 limit (3/5 seeds, flips with seed ,  marginal)

The engine was tuned, historically, to the Fed lognormal; closing this gap now
means tuning toward a measured distribution, and the amount distance is the
largest single fidelity defect the project has. That is a generative-modeling
slice, not a gate-tuning slice: the bar is set by data, so the only direction
is the generator's.

### Update 2026-10-16: the amount gap was closed, both continuous gates pass

The generative-modeling work landed: the spec/02 ticket recalibration (mu_log +0.90, every
ticket quantile x e^0.90 ~ 2.46) moved the synthetic amounts onto the real corpus's scale,
and the card-roster density revision recorded in `spec/18` §4 changed the arrival sample.
Re-measured at the pinned protocol (`prism_generator benchmark -n 2000 --unified --seed 42`):

| corpus | amount W1 | limit | arrival W1 | limit | verdict |
|---|---|---|---|---|---|
| Prism Generator synthetic, n=2,000 (protocol seed 42) | 0.0650 | 0.0916 PASS | 0.0345 | 0.0446 PASS | PASS |
| across the five stability seeds | 0.0421  to  0.0650 | PASS 5/5 | 0.0296  to  0.0397 | PASS 5/5 | PASS |

The 2026-10-15 row above (0.3732 FAIL / 0.0457 FLIP) is historical. Both continuous gates
now pass on every pinned seed, and the pinned run grades `TIER-1_GOLD` with 0 violations;
the only Pillar-1 mover left is MCC JS (2/5 seeds), which is a merchant-mix statistic, not
an amount or arrival one.
