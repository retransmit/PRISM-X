# Prism

Real-time fraud decisioning for card issuers, with every alert explained.
Prism scores an authorisation request before the issuer answers, keeps an
audit record of every decision, lets an analyst and a cardholder respond, and
is evaluated the way a bank would evaluate it: leakage-free, time-ordered where
the data is a stream, at a false-positive budget, with the evidence behind
every number committed beside it.

## What it does

- Scores each request with the champion model on the request path and
  answers; the challengers' scores and the alert's explanation follow onto the
  audit row (`src/prism/api/main.py`).
- Explains every alert from measured facts: where the amount sits among all
  transactions, the daily cycle, which characteristics are out of range, how
  many of the most similar transactions were fraud, and the amount at which it
  would have cleared, stated only as far as the sweep found it true
  (`src/prism/explain/narrative.py`). No anonymised component is ever named.
- Keeps an append-only audit log with unique transaction ids: a retried
  request is answered from the log, a different request under a used id is
  refused (`src/prism/api/db.py`).
- Takes verdicts only with a credential, one per role, and records the role
  the credential names (`api.credentials`).
- Trains, tunes and evaluates under rules a test asserts: split before any
  fit, resampling inside training folds only, thresholds from validation, the
  test window scored once, latency judged on p99 at the HTTP boundary.
- Records every deployment in a manifest: each model's threshold and the
  preprocessing it was fitted under. A partial retrain that would leave a
  model behind new preprocessing is refused before anything is written; a
  model without a tuned threshold is refused at startup
  (`src/prism/training/manifest.py`).
- Audits every feature before training for what it alone gives away about
  the label, and refuses a feature above the track's ceiling unless the
  configuration allows it with a reason
  (`reports/metrics/<track>/shortcut_audit.json`).

Three detectors run behind one interface: an Isolation Forest, a PyTorch
autoencoder trained on legitimate rows, and a cost-weighted XGBoost. Each is
explained in its own exact terms (XGBoost's TreeSHAP, tree SHAP on the
Isolation Forest's path length, the autoencoder's per-feature reconstruction
error), every explanation ends in reason codes per driver family, and the
champion is chosen on validation PR-AUC and recorded in the bundle manifest.

## Two data tracks

- **ULB**: 284,807 real European card transactions from 2013, anonymised into
  principal components, one fraud in 578 rows. Real data; no feature names.
- **India**: a synthetic card stream from the transaction generator,
  calibrated to the Reserve Bank of India's published aggregates, with named
  pre-authorisation features, card velocity features computed point in time,
  and the card's limit entering as ratios rather than as the identifying raw
  value. Evaluated in time order at a false-positive budget, with adjusted
  metrics reported over the registry's range of all-payments fraud rates.

## Setup

Python 3.14 (the autoencoder is PyTorch, which has a native wheel; TensorFlow
publishes none for this interpreter).

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Download the dataset (`creditcard.csv`, 284,807 rows / 492 fraud) from Kaggle
"Credit Card Fraud Detection" (ULB) and drop it in `data/raw/`. It is gitignored, so never
commit it.

The Indian track reads a synthetic export under `data/raw/india/` (gitignored).
The transaction generator ships in this repository under `generator/` and is
installed by the requirements as `prism-generate`. Generate the export with it,
for training at a stated boost and once more for the demo replay:

    prism-generate generate -n 90000 -c 2000 -m 300 --region IN --calibration rbi-psi-2026-07 --fraud-rate 0.005 --days 60 --seed 42 -o data/raw/india/train_in.csv --export-institutional-views
    prism-generate generate -n 20000 -c 600 -m 150 --region IN --calibration rbi-psi-2026-07 --fraud-rate 0.03 --days 14 --seed 7 -o data/raw/india/demo_in.csv --export-institutional-views

Keep `-n` below 100,000: above it the generator's CLI switches to its parallel
coordinator, which writes a directory and takes neither `--calibration` nor
the institutional views. Each run writes `<stem>_calibration.json`; the report
states the registry rate, the requested rate and the boost factor.

Copy `config.local.yaml.example` to `config.local.yaml` for a deployment: the
track it serves, its audit log and the verdict credentials. `PRISM_TRACK` and
`PRISM_DATABASE_URL` in the environment override the same two settings.

## Running

The pipeline, per track (`--track ulb` or `--track india`; without the flag
the `track` key in `config.yaml` or `config.local.yaml` applies):

```bash
python -m prism.data.eda --track ulb                        # dataset figures and statistics (ULB)
python -m prism.training.train --track india --model all    # shortcut audit, train, tune, manifest
python -m prism.evaluation.experiments --track india        # resampling against weighting, with a paired interval
python -m prism.evaluation.benchmark --track india          # the deployed models on the test window
python -m prism.evaluation.latency --track india --explain --http   # engine, explanation and HTTP round trip
python -m prism.harness.generate                            # the harness's seed streams, from the shipped generator
python -m prism.harness.run                                 # rolling-origin backtests, label regimes, novel attacks, intervals
python -m prism.reporting.results --readme README.md       # the Results sections
```

Artifacts live under `models/<track>/` (models, scaler, encoder, explanation
reference, `manifest.json`) and under `reports/metrics/<track>/`, which is
committed. The service, the replay and the dashboard:

```bash
uvicorn prism.api.main:app                       # service on :8000, serving the configured track
python simulator/stream.py --track india --rate 1.5   # replays data/raw/india/demo_in in time order
cd frontend && npm run dev                       # dashboard on :5173
```

The live loop (Indian track): a four-month stream no model has seen, replayed through
the service while the runner plays the bank. The service runs on its own audit log,
and the runner posts labels with the `label_feed` credential:

```bash
python -m prism.live.generate                                   # the live stream, from the shipped generator
PRISM_LIVE=1 PRISM_TRACK=india PRISM_DATABASE_URL=sqlite:///prism_audit_live.db uvicorn prism.api.main:app
PRISM_LABEL_TOKEN=<api.credentials.label_feed> python -m prism.live.run --endpoint http://127.0.0.1:8000
python -m prism.live.retrain --as-of <stream seconds>           # one retrain by hand; the runner calls it itself
```

Each retrain writes a bundle under `models/live/india/bundles/`; a promotion rewrites
`models/live/india/current.json`, which a service started with `PRISM_LIVE=1` polls and
swaps to once the new bundle has loaded. Any other service, the benchmarks included,
serves `models/india/` and never reads the pointer. A bundle or a pointer that cannot be
loaded leaves the serving engine in place and says why on `/health`; rewriting the pointer
tries it again, so a repaired bundle can be promoted under the same id. To roll back, write
`{"bundle_id": "offline"}` (or an earlier bundle's id) to the pointer, or delete it. A retrain
refuses a training window the shortcut audit flags, and keeps its audit and its backtest
scores in the bundle.

A run starts clean: the runner refuses a live model root that holds an earlier run (move
`models/live/india/` aside) and a service whose audit log already holds payments, and a
schedule whose first retrain would have no training window. A payment the service fails
to answer counts as a failed request and the stream goes on. The runner stops, with its
log written and marked incomplete, when a retrain read no rows (its `--database-url` is
not the service's audit log) or when the service does not serve a promoted bundle after
`live.swap_checks` health checks.

`python -m prism.training.train --model <name>` retrains one model; the
manifest keeps the others' thresholds and refuses the run if their
preprocessing would no longer match. `/health` and `/metrics` say which track
the service serves, and `/health` gives the reason when no models are served.

Upgrading a deployment that predates these rules: an audit log with duplicate
transaction ids refuses to upgrade and names them with a count, so point
`api.database_url` at a new log or remove the duplicates first; rows logged
before the channel type was stored get it back-filled from the display channel;
a model directory without a manifest is adopted when its scaler and encoder are
the ones a fresh fit produces, and otherwise needs `--model all`; a model
calibrated under the earlier score map is refused with the instruction to
retrain it.

The service exposes ten endpoints under `/api/v1`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | readiness, which models are loaded, which track |
| POST | `/predict` | score one transaction with the champion; the explanation lands on its audit row |
| POST | `/explain` | attribution for any transaction, synchronously |
| GET | `/metrics` | thresholds, benchmark table, PR curves, which track |
| GET | `/transactions` | audit-log rows, newest first; `flagged=` and `card_id=` filters |
| POST | `/decisions` | an analyst verdict or a cardholder response, with a bearer credential; append-only |
| GET | `/stats` | counters from the audit log: throughput, p99 latency, decisions, cards |
| POST | `/labels` | a label as the bank learns it (chargeback, investigator verdict, clean maturity), with the label feed's credential; Indian track |
| GET | `/live` | the serving bundle and its champion, labels received by source, the last retrains; Indian track |
| GET | `/drift` | input and score drift of the serving bundle against its own training reference; Indian track |

A transaction carries the active track's inputs (`Time, V1..V28, Amount` on ULB;
the pre-authorisation columns plus `tx_time_seconds`, `card_id`, `credit_limit`
and `available_balance` on India) and may carry `merchant`, `channel` and
`currency` for the boards. On the Indian track the service derives the limit
ratios and the velocity features from the card's earlier rows in the audit log
before scoring. On the Indian track the request is also checked against the
shipped generator's contract (`generator/spec/19_detector_contract.yaml`): a
wrong type or a value outside a closed code set (channel, entry mode, ECI, the
3-D Secure result, AVS, ASN type, currency) is a 422 naming the field and the
value, while an unseen acquirer, gateway or product still scores. A repeated
`transaction_id` with the same request is answered from the log; with a
different request it is a 409. The latency each answer
reports is the service time from request to audit write; the benchmark's
`latency_http.json` is the round trip a caller sees against a server process.

## Layout

```
src/prism/
  config.py          config.yaml loader, local overlay, environment overrides
  tracks.py          the two tracks: columns, split rule, artifact directories
  data/              loading, schema validation, leakage-free split, encoder and scaler
  features/          velocity features from the card's earlier rows; limit ratios
  models/            three detectors behind one interface (base.py)
  training/          training entrypoint, the manifest, in-fold-only resampling
  evaluation/        PR-AUC metrics, threshold tuning, benchmark, latency, shortcut audit
  explain/           SHAP engine, measured-facts narrative, reference statistics
  api/               FastAPI service and audit-log persistence
  reporting/         the Results sections, generated from the metric files
simulator/           streaming replay client
frontend/            React dashboard (Vite, React, TypeScript)
generator/           the transaction generator: package, specifications, tests, scripts
```

## The generator

`generator/` is the transaction generator in full: the `prism_generator` package,
its specification files under `generator/spec/` (the leakage gate, the calibration
profiles, the Indian rails), its own test suite, its scripts and its research
notes. It is developed on its own line and mirrored here at a recorded revision
(`generator/SOURCE`); the copy is refreshed as a whole, never edited in place, so
a fix goes to the generator's line first and arrives here with the next refresh.
`tests/test_generator_package.py` pins what Prism relies on: the revision is
recorded, the package imports, its export contract is the one the Indian track
reads, and a small batch generates end to end.

```bash
pip install -e generator/                   # done by requirements.txt
prism-generate generate -n 5000 --region IN --calibration rbi-psi-2026-07 --fraud-rate 0.03 -o data/demo.csv --export-institutional-views
python -m prism_generator.stream --region IN --calibration rbi-psi-2026-07 --duration 60 --tps 5 --endpoint http://localhost:8000/api/v1/predict --label-file data/labels.jsonl
cd generator && python -m pytest -q         # the generator's own suite, about five minutes
```

## Dashboard

The React dashboard in `frontend/` polls the service and posts verdicts, each
with the role's credential. Three boards, one per reader of a scored
transaction: Overall (feed, live log, measured comparison), Analyst (alert
queue, evidence, two actions) and Customer (a held payment, one sentence,
"this was me" or "not me").

```bash
cd frontend
npm install
npm run dev      # dashboard on :5173
npm test         # the suite prints its count
npm run build    # production bundle
```

Verdicts need a credential. The service reads one bearer token per role from
`api.credentials` (set in `config.local.yaml`). The dashboard sends the token
stored in the browser under the `localStorage` keys `prism.credential.analyst`
and `prism.credential.customer`, which is how a shared build is used: each
person sets their own role's token once. A token can also be built into the
bundle as `VITE_ANALYST_TOKEN` or `VITE_CUSTOMER_TOKEN`, but a bundle is
readable by anyone who can open the page, so that is for a build only its
owner can reach, never for one shared with others.

A full demo needs three terminals, in this order:

```bash
.venv/Scripts/python.exe -m uvicorn prism.api.main:app              # 1. service on :8000
.venv/Scripts/python.exe simulator/stream.py --track india --rate 1.5   # 2. replay the demo export
cd frontend && npm run dev                                          # 3. dashboard on :5173
```

The service must be running before the dashboard is useful; with it down, the
dashboard says so and prints the command to start it. Fonts are self-hosted via
`@fontsource` rather than loaded from a CDN, so the demo does not depend on the
room's wifi. Risk colours are cyan and rose, not red and green, because
red/green is the worst pairing for colour-vision deficiency; risk is never
carried by colour alone, every indication also has a marker shape, a
proportional band width and the printed score.

Four steps, about four minutes, against a running service and replay:

1. Overall: the feed and log are moving; the rail shows scored, flagged, flag rate, p99 latency.
2. An alert fires. Analyst: it is at the top of the queue, status Open. Read the evidence, click Confirm fraud.
3. View as cardholder: the same card's payments as a person sees them; the held one carries its sentence. Click Not me.
4. Overall: the log line now ends with `analyst: fraud  customer: not me`.

Every number on every board comes from the audit log or `reports/metrics`;
nothing is typed in.

## How the numbers are kept honest

- **Leakage rules, asserted by tests.** `data/preprocess.py` splits first and
  fits the encoder and the scaler on training rows only; resampling happens
  inside training folds and never touches validation or test rows; thresholds
  are tuned on validation and the test window is scored once; the Indian split
  is time-ordered. `tests/test_no_leakage.py` and `tests/test_cross_validation.py`
  pin each rule.
- **The manifest.** `models/<track>/manifest.json` records every model's
  threshold, the fingerprint of the preprocessing it was fitted under and the
  digest of its file; the service refuses a model the manifest does not vouch
  for, a missing or replaced model file, and a configured champion the manifest
  does not list, and a one-model retrain can neither reset the other thresholds
  nor leave a model behind new preprocessing. A directory from before the
  manifest still serves, with every model required to have a tuned threshold
  and a startup line saying that its consistency is unverified.
- **The shortcut audit.** Before training, every feature's single-column
  ranking power and its one-class-only values are written beside the metrics;
  on the Indian track a column above the ceiling refuses training unless the
  configuration allows it with a reason.
- **Committed evidence.** The metric files under `reports/metrics/` travel
  with the repository, and `tests/test_results_generator.py` fails when the
  README's Results section does not carry every number and table the generator
  prints from them, and skips rather than passes when they are missing.
- **Explanations say only what was measured.** No anonymised component is
  named; the amount counterfactual claims a range only when every lower amount
  was found to clear; the reference statistics and the nearest-case index are
  fitted on training rows only.
- **The harness.** `prism.harness` compares the models the way the design
  asks: rolling-origin backtests over several generator seeds, the encoder and
  the scaler fitted inside each training window, the threshold tuned at the
  false-positive budget on the validation window, each test window scored
  once, training labels as the bank's supervision model would have revealed
  them by the cut-off (and the full-label upper bound beside them), one attack
  scenario held out at a time, and every figure a mean over the folds with a
  bootstrap interval. `tests/test_harness_*.py` pin the windows, the label
  masks and the held-out rows; the cost rates in `config.yaml` are marked as
  hypotheses.

Two caveats the report states plainly. `Time` on the ULB track is seconds
since the dataset's first row and is a model feature; a live feed must supply
it on the same basis or the feature must be replaced before retraining. The
Indian track is synthetic: its calibration report and the generator's leakage
gate are the evidence, the registry rate it is evaluated at counts all
domestic payments rather than cards alone, and the ULB track remains the
real-data benchmark.

## Results

Every number below is copied from a generated artifact under
`reports/metrics/<track>/` or from a measurement whose method is stated beside
it. `python -m prism.reporting.results` renders the same files into
`docs/report/results.md`, and `tests/test_results_generator.py` fails when a
number or a table the generator prints is missing here. The commands under Running regenerate all of it.

<!-- results:start -->
### ULB track (real data)

Every number here is read from `reports/metrics/ulb/` by `prism.reporting.results`; nothing is typed from memory.

**Dataset** (`eda.json`): 284,807 transactions, 492 fraud, a fraud rate of 0.1727% (1 : 577.9). Median amount is 9.25 for fraud against 22.00 for legitimate transactions. At this prevalence the no-skill baseline for PR-AUC is 0.0017, not 0.5, and accuracy is not reported anywhere.

**Split** (`split.json`): 1,081 exact duplicate rows are dropped before splitting, leaving 181,584 training, 45,396 validation and 56,746 test rows (302, 76 and 95 fraud). Thresholds are tuned on validation; the test split is scored once.

**Experiment matrix** (`experiment_matrix.md`), test split:

| Arm | SMOTE | Threshold | PR-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |
|---|---|---|---|---|---|---|---|---|
| xgboost+weighted | no | 0.7166 | **0.8169** | 0.9730 | 0.7579 | 0.8521 | 23 | 1.0 |
| xgboost+smote | yes | 0.8287 | **0.8155** | 0.9722 | 0.7368 | 0.8383 | 25 | 1.0 |
| autoencoder+weighted | no | 0.9444 | **0.4533** | 0.5541 | 0.4316 | 0.4852 | 54 | 1.8 |
| isolation_forest+weighted | no | 0.7457 | **0.1143** | 0.2283 | 0.2211 | 0.2246 | 74 | 4.4 |

The two arms are not distinguishable on this split: a paired bootstrap of the PR-AUC difference (weighting minus SMOTE) runs from -0.0176 to +0.0177, and SMOTE scores at least as well in 41% of resamples.
The deployed XGBoost is the weighted arm.

**Benchmark** (`benchmark.md`), the deployed models on the test split:

| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |
|---|---|---|---|---|---|---|---|---|
| xgboost | 0.7166 | **0.8169** | 0.9681 | 0.9730 | 0.7579 | 0.8521 | 23 | 1.0 |
| autoencoder | 0.9444 | **0.4533** | 0.9374 | 0.5541 | 0.4316 | 0.4852 | 54 | 1.8 |
| isolation_forest | 0.7457 | **0.1143** | 0.9308 | 0.2283 | 0.2211 | 0.2246 | 74 | 4.4 |


**Latency** (`latency.json`), unseen rows through the scoring path, champion only, in the engine over 500 rows: mean 6.72 ms, p50 6.57 ms, p95 7.54 ms, **p99 7.80 ms** against the 100 ms budget; the single slowest request took 8.17 ms. The budget is judged on p99, so this path is within budget.

**Explanation latency** (`latency_explain.json`), the explanation alone, which runs after the answer, per row over 500 rows: mean 29.99 ms, p50 30.60 ms, p95 32.62 ms, **p99 33.37 ms** against the 100 ms budget; the single slowest request took 78.16 ms. The budget is judged on p99, so this path is within budget.

**HTTP latency** (`latency_http.json`), the round trip a caller sees against the service running in its own process, every test fraud plus a random sample in time order, not flagged, 523 requests: p50 40.48 ms, p95 63.61 ms, **p99 76.66 ms**; flagged, 72 requests: p50 41.62 ms, p95 65.92 ms, **p99 77.40 ms**; against the 100 ms budget, judged on p99, the service is within the budget.

### Indian track (synthetic export, RBI-calibrated)

Every number here is read from `reports/metrics/india/` by `prism.reporting.results`; nothing is typed from memory.

**Training export** (`exports.json`, `calibration.json`, profile `rbi-psi-2026-07`): 90,000 card authorisations over 2,000 cards, 300 merchants and 63.8 days, generated at a requested fraud prevalence of 0.5%, which is 404.2 times the registry rate of 1 in 80,847 (1.2369e-05); at the registry rate the export would hold 1.1 fraud rows, so the boost is stated beside every number below. The export comes from generator revision `51e5c77` (`generator/SOURCE`).
The report's gated target `mean_credit_card_ticket_inr` passes: the mean amount of legitimate rows on credit card products is Rs 3,225 against the published Rs 3,440, inside the 25% tolerance. Every other target is reported rather than gated.

**Split** (`split.json`), time-ordered 70/10/20 on the transaction clock: 62,999 training, 9,000 validation and 18,001 test rows (307, 39 and 89 fraud); no duplicate rows. Thresholds are tuned on the validation window to a false-positive budget of 5 alerts per 10,000 legitimate transactions (`tracks.india.fpr_budget_per_10k`), not to F1, which weighs a missed fraud and a false alert equally and moves with the prevalence, so at the registry rate its optimum sits wherever the validation window happens to show no false alert; the test window is scored once. The test window's own prevalence, 0.0049, is the no-skill baseline for the PR-AUC below, and accuracy is not reported anywhere.

**Experiment matrix** (`experiment_matrix.md`), test window:

| Arm | SMOTE | Threshold | PR-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |
|---|---|---|---|---|---|---|---|---|
| xgboost+smote | yes | 0.8507 | **0.5594** | 0.8919 | 0.3708 | 0.5238 | 56 | 1.1 |
| xgboost+weighted | no | 0.9034 | **0.5355** | 0.8182 | 0.4045 | 0.5414 | 53 | 1.2 |
| autoencoder+weighted | no | 0.9832 | **0.2835** | 1.0000 | 0.2022 | 0.3364 | 71 | 1.0 |
| isolation_forest+weighted | no | 0.7225 | **0.0244** | 0.4000 | 0.0225 | 0.0426 | 87 | 2.5 |

The two arms are not distinguishable on this split: a paired bootstrap of the PR-AUC difference (weighting minus SMOTENC) runs from -0.0763 to +0.0273, and SMOTENC scores at least as well in 83% of resamples.
The deployed XGBoost is the weighted arm.

**Benchmark** (`benchmark.md`), the deployed models on the test window:

| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch | Precision at 1 in 80,847 | FPR per 10k | Value above floor caught |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost | 0.9034 | **0.5355** | 0.9762 | 0.8182 | 0.4045 | 0.5414 | 53 | 1.2 | 0.0111 | 4.5 | 0.533 |
| autoencoder | 0.9832 | **0.2835** | 0.8184 | 1.0000 | 0.2022 | 0.3364 | 71 | 1.0 | 1.0000 | 0.0 | 0.955 |
| isolation_forest | 0.7225 | **0.0244** | 0.7484 | 0.4000 | 0.0225 | 0.0426 | 87 | 2.5 | 0.0017 | 1.7 | 0.006 |

"Precision at 1 in 80,847" is the prevalence-adjusted figure from the design (TPR times r over TPR times r plus FPR times one minus r, at r = 1.2369e-05), "FPR per 10k" is false positives per 10,000 legitimate transactions, and "Value above floor caught" is the share of fraud value above Rs 10,000 (`tracks.india.value_floor`) that the model flags. The rate is the registry's for all domestic payments, most of them UPI (the research note gives the shares); a card-only rate is not published, so it is a reference, not the card rate. At it the xgboost model's precision is 1.1% with 4.5 alerts per 10,000 legitimate transactions; that is what a false-positive budget buys at one fraud in 80,847, and the report must say so. Over the registry's monthly span, one in 121,303 to one in 47,018, that precision runs from 0.7% to 1.9%.

**Latency** (`latency.json`), unseen rows through the scoring path, champion only, in the engine over 500 rows: mean 17.60 ms, p50 17.31 ms, p95 19.06 ms, **p99 19.57 ms** against the 100 ms budget; the single slowest request took 68.66 ms. The budget is judged on p99, so this path is within budget.

**Explanation latency** (`latency_explain.json`), the explanation alone, which runs after the answer, per row over 500 rows: mean 44.42 ms, p50 44.83 ms, p95 47.66 ms, **p99 49.16 ms** against the 100 ms budget; the single slowest request took 80.78 ms. The budget is judged on p99, so this path is within budget.

**HTTP latency** (`latency_http.json`), the round trip a caller sees against the service running in its own process, every test fraud plus a random sample in time order, not flagged, 558 requests: p50 56.74 ms, p95 74.22 ms, **p99 91.44 ms**; flagged, 27 requests: p50 57.26 ms, p95 80.82 ms, **p99 88.13 ms**; against the 100 ms budget, judged on p99, the service is within the budget.

**Harness** (`harness/results.json`, `harness/summary.json`), the comparison the design asks for, on 5 seeds of the generator's stream (42, 7, 11, 23, 59), each 180,000 rows over 120 days at a requested fraud rate of 0.005: rolling-origin backtests with a training window of 60 days growing by 15 days per fold, a 10-day validation window and a 20-day test window (4 folds per seed); the encoder and the scaler are fitted inside each training window, the threshold is tuned at the false-positive budget on the validation window, and each test window is scored once. Labels come in two regimes: operational, a fraud counts as fraud only once the bank's supervision model (an investigator queue of 50 cases a day at an alert threshold of 0.7, chargebacks with their delays, fraud that is never reported) has discovered it by the window's end; full, every label known, the upper bound. Evaluation always uses the truth. Each value is the mean over the seed-by-fold runs with a 95 percent bootstrap interval of that mean. The scoring p99 is the model's own single-row call on preprocessed rows, not the service path of `latency.json`. In 36 of 60 operational runs no fraud was known in the validation window and the cutoff came from the budget alone.

| Model | Labels | Folds | Cutoffs from the budget alone | PR-AUC | Recall at budget | Precision | False alerts per 10k | Alerts per day | Cost per 10k (INR) | Scoring p99 (ms) | Explanation p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| autoencoder | full | 20 | 0 | 0.1932 [0.1657, 0.2233] | 0.144 [0.115, 0.176] | 0.782 [0.697, 0.865] | 2.4 [1.4, 3.5] | 1.2 [0.9, 1.4] | 167434 [147013, 190468] | 1.12 [1.10, 1.15] | 1.27 [1.24, 1.31] |
| autoencoder | operational | 20 | 12 | 0.1626 [0.1356, 0.1951] | 0.048 [0.028, 0.070] | 0.563 [0.378, 0.742] | 1.1 [0.4, 1.9] | 0.4 [0.3, 0.6] | 726954 [557881, 904974] | 1.10 [1.07, 1.13] | 1.31 [1.27, 1.35] |
| isolation_forest | full | 20 | 0 | 0.0179 [0.0160, 0.0199] | 0.000 [0.000, 0.001] | 0.005 [0.000, 0.015] | 0.5 [0.0, 1.0] | 0.1 [0.0, 0.1] | 1111832 [943108, 1291721] | 18.22 [17.63, 18.89] | 4.36 [4.20, 4.58] |
| isolation_forest | operational | 20 | 12 | 0.0179 [0.0160, 0.0199] | 0.007 [0.003, 0.010] | 0.061 [0.031, 0.097] | 3.4 [1.9, 5.0] | 0.5 [0.3, 0.7] | 1081575 [914816, 1261190] | 17.85 [17.44, 18.24] | 4.48 [4.27, 4.72] |
| xgboost | full | 20 | 0 | 0.5704 [0.5456, 0.5917] | 0.408 [0.381, 0.435] | 0.820 [0.780, 0.857] | 4.6 [3.5, 5.6] | 3.0 [2.8, 3.3] | 325005 [271965, 379673] | 10.80 [10.57, 11.06] | 12.78 [12.60, 12.99] |
| xgboost | operational | 20 | 12 | 0.2679 [0.2354, 0.3015] | 0.073 [0.047, 0.099] | 0.654 [0.472, 0.825] | 0.9 [0.2, 2.0] | 0.6 [0.3, 0.8] | 747451 [629272, 872084] | 10.61 [10.47, 10.76] | 12.45 [12.29, 12.62] |

Under operational labels, the regime the bank would have had, the best model is the xgboost at PR-AUC 0.2679 [0.2354, 0.3015] with recall at the budget 0.073 [0.047, 0.099]; with every label known its PR-AUC is 0.5704 [0.5456, 0.5917] and its recall 0.408 [0.381, 0.435]: the price of learning from what the bank knew at the time.
Under operational labels, against the autoencoder the difference in PR-AUC on the 20 folds both ran is 0.11 [0.07, 0.14] on the xgboost's side, which is at least as good on 85% of them.
Under operational labels, against the isolation_forest the difference in PR-AUC on the 20 folds both ran is 0.25 [0.22, 0.28] on the xgboost's side, which is at least as good on 100% of them.

Novel attacks (`harness/scenario.json`): on each seed's last fold, one fraud scenario at a time has its labelled rows removed from the training and validation windows (the card histories the velocity features count keep them) while the test window keeps every row, so the held-out recall is the share of that scenario's fraud rows the model still catches at its budget threshold, averaged over the seeds.

| Model | Held-out scenario | Seeds | Held-out fraud rows | Held-out recall | PR-AUC |
|---|---|---|---|---|---|
| autoencoder | INTENT_OMEGA_BISECT_DRAIN | 5 | 166 | 0.137 [0.064, 0.196] | 0.1705 [0.1420, 0.1908] |
| autoencoder | INTENT_OMEGA_HARVEST | 5 | 165 | 0.152 [0.035, 0.268] | 0.1638 [0.1351, 0.1874] |
| autoencoder | INTENT_OMEGA_PROBE | 5 | 252 | 0.004 [0.000, 0.013] | 0.1705 [0.1427, 0.1930] |
| isolation_forest | INTENT_OMEGA_BISECT_DRAIN | 5 | 166 | 0.000 [0.000, 0.000] | 0.0169 [0.0134, 0.0202] |
| isolation_forest | INTENT_OMEGA_HARVEST | 5 | 165 | 0.000 [0.000, 0.000] | 0.0171 [0.0127, 0.0216] |
| isolation_forest | INTENT_OMEGA_PROBE | 5 | 252 | 0.000 [0.000, 0.000] | 0.0174 [0.0131, 0.0224] |
| xgboost | INTENT_OMEGA_BISECT_DRAIN | 5 | 166 | 0.165 [0.076, 0.245] | 0.5341 [0.4958, 0.5631] |
| xgboost | INTENT_OMEGA_HARVEST | 5 | 165 | 0.196 [0.117, 0.263] | 0.5008 [0.4713, 0.5312] |
| xgboost | INTENT_OMEGA_PROBE | 5 | 252 | 0.008 [0.000, 0.016] | 0.3363 [0.2800, 0.3862] |

Trained without INTENT_OMEGA_PROBE, the autoencoder catches 0.004 [0.000, 0.013] of its rows in the test window (252 held-out fraud rows over 5 seeds), its weakest novel attack.
Trained without INTENT_OMEGA_BISECT_DRAIN, the isolation_forest catches 0.000 [0.000, 0.000] of its rows in the test window (166 held-out fraud rows over 5 seeds), its weakest novel attack.
Trained without INTENT_OMEGA_PROBE, the xgboost catches 0.008 [0.000, 0.016] of its rows in the test window (252 held-out fraud rows over 5 seeds), its weakest novel attack.

The supervision model's timelines, which the operational labels follow: seed 42, 899 fraud rows, 6 discovered by investigators, 506 by chargebacks, 387 never reported, median discovery 29.2 days after the payment; seed 7, 871 fraud rows, 6 discovered by investigators, 444 by chargebacks, 421 never reported, median discovery 28.3 days after the payment; seed 11, 901 fraud rows, 6 discovered by investigators, 531 by chargebacks, 364 never reported, median discovery 30.8 days after the payment; seed 23, 844 fraud rows, 1 discovered by investigators, 495 by chargebacks, 348 never reported, median discovery 29.0 days after the payment; seed 59, 855 fraud rows, 6 discovered by investigators, 464 by chargebacks, 385 never reported, median discovery 29.6 days after the payment.

The cost column counts a missed fraud at its amount, a review at 150 INR per alert and a false decline at 60 INR of friction per false positive, per 10,000 transactions; the two rates are hypotheses in `config.yaml` until an issuer's figures replace them. The streams were written by the shipped generator at revision `51e5c77c0028`.

**Live loop** (`live/run.json`, `live/events.json`): a stream the models never saw, written by the shipped generator at revision 51e5c77c00280734e46097f4ebd872ab887f3783 (seed 101, 180,000 rows requested over 120 days, 2,000 cards, 300 merchants, fraud rate 0.005), was replayed through the service in time order: 180,000 payments over 137 stream days, 853 of them fraud, 0 refused. The bank was the generator's supervision model: its investigators reviewed Prism's alerts and nothing else, up to 50 a day; chargebacks arrived at their own delays; unreported fraud never did. A challenger was trained every 15 days from day 60 on the labels known by then, backtested against the serving champion on a 20-day window ending 30 days before the clock, and promoted only when the 95 percent paired interval of the PR-AUC difference (2,000 resamples) lay above zero.

Labels the bank learned and sent: 290 chargebacks, 371 investigator verdicts, 65,114 clean maturities; 221 fraud rows were never reported and 114,004 labels were still to come when the stream ended; 0 labels were refused.

Retrains: 0 of 6 retrains promoted.

| Stream day | Verdict | Challenger PR-AUC | Champion PR-AUC | Paired interval | Challenger, truth | Champion, truth | Why |
|---|---|---|---|---|---|---|---|
| 60 | skipped | n/a | n/a | n/a | n/a | n/a | the training window holds 0 known frauds among 0 rows, fewer than min_known_fraud 10 |
| 75 | kept | 0.1722 | 0.4478 | [-0.3778, -0.1729] | 0.1860 | 0.3699 | the paired interval of the PR-AUC difference [-0.3778, -0.1729] does not lie above zero |
| 90 | kept | 0.2129 | 0.6132 | [-0.5196, -0.2775] | 0.1535 | 0.5145 | the paired interval of the PR-AUC difference [-0.5196, -0.2775] does not lie above zero |
| 105 | kept | 0.5297 | 0.7054 | [-0.2567, -0.0963] | 0.4446 | 0.5468 | the paired interval of the PR-AUC difference [-0.2567, -0.0963] does not lie above zero |
| 120 | kept | 0.6157 | 0.5941 | [-0.0631, 0.1107] | 0.5108 | 0.4905 | the paired interval of the PR-AUC difference [-0.0631, 0.1107] does not lie above zero |
| 135 | kept | 0.5090 | 0.4620 | [-0.0193, 0.1168] | 0.4577 | 0.4018 | the paired interval of the PR-AUC difference [-0.0193, 0.1168] does not lie above zero |

The promotion reads the operational labels: on the backtest window the incumbent's alerts were investigated within hours, while the fraud it missed waits for a chargeback or is never reported, so those labels favour the incumbent over a challenger that catches different fraud. The truth columns score the same rows against the generator's ground truth, which the runner holds and the retrain job never sees.

The served champion against the truth, per window (the runner computes these; the service never sees the truth):

| Days | Payments | Fraud | PR-AUC | Recall | Alerts | Serving |
|---|---|---|---|---|---|---|
| 0 to 15 | 22,251 | 96 | 0.4900 | 0.3125 | 33 | offline |
| 15 to 30 | 20,514 | 95 | 0.4223 | 0.2632 | 32 | offline |
| 30 to 45 | 20,240 | 89 | 0.3300 | 0.1910 | 31 | offline |
| 45 to 60 | 19,776 | 103 | 0.5598 | 0.3786 | 51 | offline |
| 60 to 75 | 20,044 | 118 | 0.5598 | 0.3729 | 56 | offline |
| 75 to 90 | 18,702 | 98 | 0.4754 | 0.3571 | 45 | offline |
| 90 to 105 | 18,937 | 80 | 0.3939 | 0.3000 | 37 | offline |
| 105 to 120 | 18,729 | 83 | 0.4453 | 0.3373 | 37 | offline |
| 120 to 135 | 18,578 | 77 | 0.4785 | 0.3766 | 40 | offline |
| 135 to 150 | 2,229 | 14 | 0.5374 | 0.5000 | 13 | offline |

Drift of the serving bundle at each retrain, over its last rows: day 60: score PSI 0.0068 (ok), highest input gateway_provider at 0.1870 (warn); day 75: score PSI 0.0021 (ok), highest input gateway_provider at 0.1790 (warn); day 90: score PSI 0.0028 (ok), highest input mcc at 0.2342 (warn); day 105: score PSI 0.0055 (ok), highest input mcc at 0.2072 (warn); day 120: score PSI 0.0045 (ok), highest input gateway_provider at 0.2220 (warn); day 135: score PSI 0.0039 (ok), highest input utilization at 0.2304 (warn).

**Demo export** (`exports.json`): 20,000 rows over 600 cards, 150 merchants and 16.0 days, 569 fraud rows at a requested 3% (2,425 times the registry rate), generated with a different seed from the training export so every row the simulator replays is unseen. Its calibration report passes the gated target `mean_credit_card_ticket_inr` (Rs 3,477 observed against Rs 3,440), so it feeds the boards and nothing in this section.
<!-- results:end -->

## License

Apache License 2.0 (`LICENSE`), covering the repository as a whole, the generator under `generator/` included.
