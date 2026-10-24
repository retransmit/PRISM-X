"""Pillar 1's real-corpus calibration reference (spec/18, `pillar_1_real_reference`).

Why this module exists
----------------------
Pillar 1's two continuous gates were re-derived here because the 2026-10-10
external-corpus scorecard (`reports/pillar1_external_corpus_scorecard.md`),
measured with the grader's own code, showed that **real card traffic fails the
hardcoded reference harder than the synthetic output does**:

    log10(1+amount)  W1: control 0.0000 PASS | ours n=1,500  0.1508 FAIL | ULB real 0.3745 FAIL

A gate that a reference-drawn control passes while a real corpus fails harder
than the generator being graded does not measure realism -- it measures distance
to a hardcoded `lognormal(mu=3.75, sigma=0.85)` USD draw.

On 2026-10-15 the reference was upgraded from the last resort (ULB, September
2013) to the corpus the retrieval had previously blocked out: the IEEE-CIS /
Vesta competition training set (`spec/18 pillar_1_real_reference`) -- 590,540
real card-not-present transactions gathered by Vesta, covering 182 days at
1-second timestamp resolution (measured median inter-arrival 14 s), amounts in
USD, with a measured 3.50% fraud rate -- fetched through the Kaggle competition
API from the project owner's own joined account after the kagglehub anonymous
path was re-measured to fail. That removes the three ULB limitations the owner
objected to (age, currency reconciliation need, whole-second clock mass).

The gates after this change
---------------------------
1. Amount realism gate: `W1(log10(1+amount)_synthetic, log10(1+amount)_real)`
   where the real side is a synthetic-independent deterministic draw from the
   corpus's legitimate rows converted to USD at a recorded constant (1.0 for the
   USD-denominated IEEE-CIS corpus), and the limit is the spec-pinned quantile
   of bootstrap real-to-real W1 at the same sample size. A corpus of real
   traffic graded against itself therefore passes at ~quantile rate; anything
   distant from real traffic fails.
2. Arrival realism gate: `W1(log10(1+dt/median_dt)_synthetic,
   log10(1+dt/median_dt)_real)`. Both sides are normalised by their own median
   inter-arrival, which makes the statistic rate-invariant: comparing a 0.038
   tx/s real window against a 0.002 tx/s protocol stream would otherwise
   measure the rate, not the arrival shape. The limit uses the same
   size-corrected real-to-real bootstrap quantile.

Known limits, stated rather than hidden
---------------------------------------
- The reference is one decade's single collection: real CNP transactions across
  Vesta's merchant base over 182 days, with only `TransactionDT`,
  `TransactionAmt` and `isFraud` used. It is not a card-rail panel with
  geographic or issuer breakdowns, so the gates are directional (distance to
  this real corpus), and the scorecard prints the corpus identity alongside
  every calibrated number.
- The calibration bar is drawn from the same corpus that sets the reference, so
  it inherits that corpus's clock resolution (1-second grid, measured median dt
  14 s and minimum 1 s). A microsecond-resolution synthetic sample is compared
  against a second-gridded real one; the scorecard states this.

Fallback
--------
If the corpus file, the spec section, or the sha256 check disagrees, callers
fall back to the previous hardcoded-reference behaviour
(`reference_note` says so by name), so packaged installs and corpora-free
environments keep working. The fallback is never silent: the scorecard and the
Markdown report always say which reference was used.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.stats

# One calibration seed for every reference draw and every bootstrap, independent
# of the run seed, so a graded run's metric is compared against the same
# reference sample across the five protocol seeds and the reported limit does
# not move with the seed under test.
CALIBRATION_SEED = 20261015


def _find_repo_root() -> Path:
    """The package's repository root: the nearest ancestor holding `spec/`.

    Mirrors `spec_loader._find_spec_dir`'s upward walk so the module keeps
    working when the package is imported from an installed position.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "spec").is_dir():
            return parent
    raise FileNotFoundError("no ancestor directory contains a spec/ folder")


def _load_reference_config() -> Dict[str, Any]:
    """The `pillar_1_real_reference` section of spec/18, or an empty dict."""
    try:
        import yaml

        from .spec_loader import _find_spec_dir

        spec_path = _find_spec_dir() / "18_benchmark_reporting.yaml"
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        body = raw.get("pillar_1_real_reference") or {}
        return dict(body) if isinstance(body, dict) else {}
    except Exception:
        return {}


def _sha256(path: Path) -> str:
    """Streaming sha256 so a ~680 MB corpus does not have to be held in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_by_own_median(deltas: np.ndarray) -> np.ndarray:
    """The graded side's exact transform: divide by the sample's own median and
    take log10(1 + x), which makes the arrival statistic rate-invariant."""
    md = float(np.median(deltas))
    if not np.isfinite(md) or md <= 0.0:
        md = 1.0
    return np.log10(1.0 + deltas / md)


def _load_corpus_table(path: Path, kind: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three reference columns -- time, amount, label -- from the configured corpus.

    `kind` names the loader:
      - `"csv"`: the IEEE-CIS `train_transaction.csv` schema (`TransactionDT`,
        `TransactionAmt`, `isFraud`), read through polars so only three of the
        394 columns ever occupy memory.
      - `"arff"`: the ULB `creditcard.arff` schema (`Time`, `Amount`, `Class`),
        kept because the corpus research of 2026-10-15 still uses it for the
        comparative scorecard.
    """
    if kind == "csv":
        import polars as pl

        frame = pl.read_csv(
            path,
            columns=["TransactionDT", "TransactionAmt", "isFraud"],
            schema_overrides={"TransactionDT": pl.Int64, "TransactionAmt": pl.Float64, "isFraud": pl.Int64},
        )
        times = frame["TransactionDT"].to_numpy().astype(np.float64)
        amounts = frame["TransactionAmt"].to_numpy().astype(np.float64)
        labels = frame["isFraud"].to_numpy().astype(np.float64)
        return times, amounts, labels
    if kind == "arff":
        from scipy.io import arff

        data, meta = arff.loadarff(str(path))
        names = [str(x).lower() for x in meta.names()]
        if not {"time", "amount", "class"} <= set(names):
            raise ValueError(f"{path.name} is not the ULB layout: columns are {names[:6]} ...")
        times = np.asarray(data["Time"], dtype=np.float64)
        amounts = np.asarray(data["Amount"], dtype=np.float64)
        labels = np.asarray(data["Class"], dtype=np.float64)
        return times, amounts, labels
    raise ValueError(f"pillar_1_real_reference.corpus_kind {kind!r} is not one of csv|arff")


@dataclass
class RealReference:
    """The corpus-derived reference pools and the calibration parameters.

    Every sample and limit is measured from one corpus file whose sha256 the
    spec records; `note` names the provenance so reports can quote it verbatim.
    """

    corpus_name: str
    corpus_sha256: str
    note: str
    amount_log_pool: np.ndarray
    """log10(1 + amount_usd) over the corpus's legitimate rows."""

    arrival_log_pool: np.ndarray
    """log10(1 + dt/median_dt)-positive inter-arrivals over legit-only pairs."""

    arrival_delta_sequence: np.ndarray
    """The same deltas as `arrival_log_pool`, in their original time order and
    un-normalised. The arrival limit is calibrated from contiguous slices of
    this sequence, because a graded stream's deltas come from a contiguous
    arrival process: random-row subsets of the corpus have a different
    inter-arrival shape (2026-10-15 measurement: a random 2,000-row IEEE-CIS
    subset fails its own gate at 0.0365 > 0.0207), so calibrating on iid pool
    draws would set the bar below the noise a real stream actually carries."""

    clock_note: str
    """The corpus's own timestamp-resolution caveat (e.g. IEEE-CIS is 1-second
    gridded; ULB is whole-second with many zero-length gaps). Read from the spec
    so a provenance note names the *active* corpus's clock, not a hardcoded
    one."""

    amount_value_sequence: np.ndarray
    """The legitimate amounts in their original time order. The amount limit is
    calibrated from contiguous corpus windows for the same reason as the
    arrival gate: a 182-day real stream carries segment heterogeneity (seasonal
    spend drift), and an iid-pair limit measures only sampling noise, which
    real windows then exceed (2026-10-15: 1 of 3 real n=2,000 windows failed its
    own iid-calibrated amount gate)."""

    usd_scale: float
    """The corpus-to-USD constant (`amount_usd_per_curve_unit`) the pool was built with.
    `amount_value_sequence` already carries it; this field only documents it."""

    quantile: float
    """The bootstrap quantile the calibrated limit is read at (spec-pinned)."""

    draws: int
    """How many real-to-real pairs the limit is averaged over (spec-pinned)."""

    _limit_cache: Dict[Tuple[str, int], float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    _legacy_limit_cache: Dict[Tuple[str, int], float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    """Separate cache for the superseded iid-pair limit. It must NOT share
    `_limit_cache`: `_legacy_pool_w1_limit` is kept only for review, and writing
    its value under the live `(kind, n)` key would silently replace the calibrated
    limit any caller of `w1_limit` gets back."""

    def amount_log_sample(self, n: int) -> np.ndarray:
        """One deterministic size-`n` draw from the corpus log-amount pool."""
        rng = np.random.default_rng(CALIBRATION_SEED)
        idx = rng.integers(0, len(self.amount_log_pool), size=n)
        return self.amount_log_pool[idx]

    def arrival_log_sample(self, n: int) -> np.ndarray:
        """One deterministic size-`n` iid draw from the rate-invariant arrival pool."""
        rng = np.random.default_rng(CALIBRATION_SEED)
        idx = rng.integers(0, len(self.arrival_log_pool), size=n)
        return self.arrival_log_pool[idx]

    def arrival_window_log_sample(self, n: int, draw: int) -> np.ndarray:
        """Deltas of one contiguous real window, normalised by the window's median.

        `_normalize_by_own_median` is applied exactly as the graded side applies
        it, so a window of the real corpus is transformed the same way a
        synthetic stream is.
        """
        seq = self.arrival_delta_sequence
        rng = np.random.default_rng(CALIBRATION_SEED + 999_000 + draw)
        start = int(rng.integers(0, max(1, len(seq) - n)))
        return _normalize_by_own_median(seq[start:start + n])

    def amount_window_log_sample(self, n: int, draw: int) -> np.ndarray:
        """Log amounts of one contiguous real window, exactly as the graded side transforms its own amounts."""
        seq = self.amount_value_sequence
        rng = np.random.default_rng(CALIBRATION_SEED + 888_000 + draw)
        start = int(rng.integers(0, max(1, len(seq) - n)))
        return np.log10(np.maximum(seq[start:start + n], 0.01) + 1.0)

    def _amount_w1_limit(self, n: int) -> float:
        """Amount limit from contiguous real windows against the pool reference,

        mirroring `_arrival_w1_limit` so both continuous limits cover real
        segment heterogeneity, not just iid sampling noise.
        """
        ref_rand = np.random.default_rng(CALIBRATION_SEED)
        w1s = np.empty(self.draws, dtype=np.float64)
        for i in range(self.draws):
            window = self.amount_window_log_sample(int(n), draw=i)
            idx = ref_rand.integers(0, len(self.amount_log_pool), size=len(window))
            ref = self.amount_log_pool[idx]
            w1s[i] = scipy.stats.wasserstein_distance(window, ref)
        return float(np.quantile(w1s, self.quantile))

    def w1_limit(self, kind: str, n: int) -> float:
        """The calibrated limit: bootstrap real-to-real W1 at the graded size.

        Both sides of every pair are independent size-`n` samples drawn from
        corpus-sized pools, `draws` times, with the calibration seed. W1's
        expectation falls with n, so the limit must be computed at the graded
        run's size -- this is the size correction the static 0.150 limit lacked.
        Memoised per (gate kind, n): the five-protocol-seed stability runner
        calls this once per seed at one sample size and deserves one price.
        """
        cached = self._limit_cache.get((kind, int(n)))
        if cached is not None:
            return cached
        if kind == "arrival":
            limit = self._arrival_w1_limit(int(n))
            self._limit_cache[(kind, int(n))] = limit
            return limit
        if kind == "amount":
            limit = self._amount_w1_limit(int(n))
            self._limit_cache[(kind, int(n))] = limit
            return limit
        raise KeyError(f"unknown gate kind {kind!r}")

    def _legacy_pool_w1_limit(self, kind: str, n: int) -> float:
        """Superseded iid-pair limit (kept so the calibration change is reviewable).

        Memoised in `_legacy_limit_cache`, deliberately separate from the live
        `_limit_cache`, so calling this cannot change what `w1_limit` returns.
        """
        cached = self._legacy_limit_cache.get((kind, int(n)))
        if cached is not None:
            return cached
        pool = self.amount_log_pool if kind == "amount" else self.arrival_log_pool
        rng = np.random.default_rng(CALIBRATION_SEED + (0 if kind == "amount" else 1))
        w1s = np.empty(self.draws, dtype=np.float64)
        for i in range(self.draws):
            a = pool[rng.integers(0, len(pool), size=n)]
            b = pool[rng.integers(0, len(pool), size=n)]
            w1s[i] = scipy.stats.wasserstein_distance(a, b)
        limit = float(np.quantile(w1s, self.quantile))
        self._legacy_limit_cache[(kind, int(n))] = limit
        return limit

    def _arrival_w1_limit(self, n: int) -> float:
        """Arrival limit from contiguous real windows against the pool reference.

        Every draw applies the graded estimator's exact shape: a contiguous
        slice of the ordered real deltas, normalised by its own median, against
        an iid pool draw of the same size (also pool-median-normalised)."""
        ref_rand = np.random.default_rng(CALIBRATION_SEED)
        w1s = np.empty(self.draws, dtype=np.float64)
        for i in range(self.draws):
            window = self.arrival_window_log_sample(int(n), draw=i)
            idx = ref_rand.integers(0, len(self.arrival_log_pool), size=len(window))
            ref = self.arrival_log_pool[idx]
            w1s[i] = scipy.stats.wasserstein_distance(window, ref)
        return float(np.quantile(w1s, self.quantile))


@functools.lru_cache(maxsize=1)
def load_real_reference() -> Optional[RealReference]:
    """The corpus reference verified against spec/18's recorded sha256, or None.

    Returns None (the caller then reports the embedded fallback by name) when
    the spec section, the corpus file or the hash disagree: a wrong hash must
    never silently grade against a different corpus than the spec names.
    """
    cfg = _load_reference_config()
    if not cfg:
        return None
    # The spec names the active corpus (`active_corpus`) as a key of its own
    # body, one entry per recorded corpus: the grader pins one and the others
    # stay recorded for the comparative scorecard. The calibration parameters
    # live on the parent block and are merged down.
    active = str(cfg.get("active_corpus", ""))
    body = cfg.get(active) if active else None
    if not isinstance(body, dict):
        return None
    cfg = {**cfg, **body}
    try:
        path = Path(str(cfg["corpus_path"]))
        if not path.is_absolute():
            try:
                path = _find_repo_root() / path
            except FileNotFoundError:
                return None
        if not path.exists():
            return None
        if _sha256(path) != str(cfg["corpus_sha256"]):
            return None

        times, amounts, labels = _load_corpus_table(
            path, str(cfg.get("corpus_kind", "arff"))
        )
        legit = labels == 0 if np.any(labels != 0) else np.ones_like(labels, dtype=bool)

        usd_per_unit = float(cfg.get("amount_usd_per_curve_unit", 1.0))
        amounts_usd = amounts[legit] * usd_per_unit
        amount_pool = np.log10(np.maximum(amounts_usd, 0.01) + 1.0)

        times_sorted = times[np.argsort(times, kind="stable")]
        labels_sorted = labels[np.argsort(times, kind="stable")]
        deltas = np.diff(times_sorted)
        positive = deltas > 0
        if bool(cfg.get("legit_only_deltas", True)):
            positive &= (labels_sorted[:-1] == 0) & (labels_sorted[1:] == 0)
        raw_deltas = deltas[positive]
        if raw_deltas.size == 0:
            return None
        median_delta = float(np.median(raw_deltas))
        if not np.isfinite(median_delta) or median_delta <= 0.0:
            return None
        arrival_pool = _normalize_by_own_median(raw_deltas)

        provenance = str(cfg.get("provenance_note", "")).strip()
        clock_note = str(cfg.get("clock_resolution_note", "")).strip()
        note = (
            f"{cfg['corpus_name']}: {int(legit.sum())} legit rows, "
            f"amounts USD-scaled at {usd_per_unit}, "
            f"arrival pool n={int(raw_deltas.size)}, median dt={median_delta:.1f} s"
            + (f"; {provenance}" if provenance else "")
            + (f"; clock: {clock_note}" if clock_note else "")
        )
        return RealReference(
            corpus_name=str(cfg["corpus_name"]),
            corpus_sha256=str(cfg["corpus_sha256"]),
            note=note,
            clock_note=clock_note,
            amount_log_pool=amount_pool,
            arrival_log_pool=arrival_pool,
            arrival_delta_sequence=raw_deltas,
            amount_value_sequence=amounts_usd,
            usd_scale=usd_per_unit,
            quantile=float(cfg.get("calibration_quantile", 0.99)),
            draws=int(cfg.get("calibration_draws", 200)),
        )
    except Exception:
        # A missing optional dependency (polars/scipy.io), a malformed corpus, an
        # unreadable spec: all degrade to the embedded fallback, which the
        # caller reports by name.
        return None
