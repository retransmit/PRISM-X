"""Module 5: the multi-model inference engine (§6.2.2).

Models, scaler and tuned thresholds are loaded once at process start and held
in memory. Loading per request would put joblib deserialisation and a torch
graph rebuild on the hot path and blow the <100ms latency target (§9) by orders
of magnitude.

One engine serves one track (design D1). Its schema names the input columns,
the categorical columns the persisted encoder maps, the columns the service
derives itself before scoring (the velocity features, design D3) and the
column order the models were trained on.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from prism import config, tracks
from prism.data.preprocess import load_encoder, load_scaler, transform_features
from prism.explain import artifacts
from prism.explain.cases import CaseIndex
from prism.explain.narrative import driver_sentence, plain_english
from prism.explain.reference import ReferenceStats
from prism.explain.shap_engine import Explanation, ShapEngine
from prism.features import ratios, velocity
from prism.models import load_all
from prism.models.base import FraudModel
from prism.training import manifest

# Kept for callers that predate the track registry; the engine reads its own schema.
FEATURE_ORDER = tracks.ULB.feature_columns

# The card's earlier transactions as (time_seconds, amount, mcc, channel).
History = Sequence[tuple[float, float, str, str]]


class InferenceEngine:
    """Holds every loaded artifact and scores one transaction at a time."""

    def __init__(self, model_dir: Path | None = None, track: str | None = None) -> None:
        self.schema = tracks.get(track) if track else tracks.active()
        self.model_dir = model_dir or self.schema.model_dir
        self.feature_order = list(self.schema.feature_columns)
        self.scale_columns = list(self.schema.scale_columns)
        self.models: dict[str, FraudModel] = {}
        self.thresholds: dict[str, float] = {}
        self.scaler = None
        self.encoder = None
        self.explainer: ShapEngine | None = None
        self.reference: ReferenceStats | None = None
        self.cases: CaseIndex | None = None
        self.warmed_up = False
        # "auto" is resolved at load from the manifest's champion.
        self.primary = config.get("api.primary_model", "auto")
        self._latencies: deque[float] = deque(
            maxlen=int(config.get("api.stats_window", 1000))
        )

    def load(self) -> "InferenceEngine":
        self.models = load_all(self.model_dir)
        if not self.models:
            raise RuntimeError(
                f"No models in {self.model_dir}. "
                f"Run `python -m prism.training.train --track {self.schema.name}`."
            )
        # Scaler and encoder must come from the same directory as the models.
        # Falling back to the config path here would silently pair one run's
        # models with another run's scaling statistics.
        self.scaler = load_scaler(self.model_dir / "scaler.joblib")
        self.encoder = load_encoder(self.model_dir / "encoder.joblib")
        if self.schema.categorical_columns and self.encoder is None:
            raise RuntimeError(
                f"No encoder in {self.model_dir}. "
                f"Run `python -m prism.training.train --track {self.schema.name}`."
            )

        record = manifest.load(self.model_dir)
        if record is not None:
            self.thresholds = manifest.thresholds(record)
            expected = record["preprocessing"]["fingerprint"]
            # The manifest names the preprocessing the models were fitted under;
            # the files on disk must be those, not a later scaler behind a stale record.
            on_disk = manifest.preprocessing_fingerprint(self.scaler, self.encoder)
            if on_disk != expected:
                raise RuntimeError(
                    f"The scaler or encoder in {self.model_dir} is not the preprocessing "
                    f"{manifest.FILENAME} records; the models cannot be served on it. Retrain with --model all."
                )
            foreign = sorted(
                name for name in self.models
                if record["models"].get(name, {}).get("fingerprint") != expected
            )
            if foreign:
                raise RuntimeError(
                    f"{self.model_dir / manifest.FILENAME} does not vouch for {', '.join(foreign)}: "
                    "fitted under other preprocessing or never recorded. Retrain with --model all."
                )
            # Every model the manifest lists must be here and be the file it
            # recorded; a missing champion must not promote a challenger, and
            # a model saved by an interrupted run must not keep an old threshold.
            absent = sorted(name for name in record["models"] if name not in self.models)
            if absent:
                raise RuntimeError(
                    f"{self.model_dir / manifest.FILENAME} lists {', '.join(absent)} but the directory cannot "
                    "load it. Restore the file or retrain with --model all."
                )
            digests = manifest.artifact_digests(self.model_dir, list(record["models"]))
            changed = sorted(
                name for name, entry in record["models"].items()
                if entry.get("sha256") and digests.get(name, {}).get("sha256") != entry["sha256"]
            )
            if changed:
                raise RuntimeError(
                    f"The file of {', '.join(changed)} in {self.model_dir} is not the one "
                    f"{manifest.FILENAME} records; its threshold does not apply. Retrain it."
                )
            if self.primary == "auto":
                # The champion chosen on validation, or the primary a manifest from before
                # champions recorded; never a guess.
                champion = (record.get("champion") or {}).get("model") or record.get("primary_model")
                if not champion or champion == "auto":
                    raise RuntimeError(
                        f"{self.model_dir / manifest.FILENAME} records no champion and api.primary_model is auto; "
                        "retrain with --model all or name a model in api.primary_model."
                    )
                print(f"[inference] primary model: {champion!r}, the champion {manifest.FILENAME} records")
                self.primary = champion
            if self.primary not in record["models"]:
                raise RuntimeError(
                    f"The configured primary model {self.primary!r} is not in {manifest.FILENAME}; "
                    "no other model is served in its place. Train it or change api.primary_model."
                )
        else:
            # A directory trained before the manifest existed. Thresholds are
            # still required; whether the models match the preprocessing on
            # disk cannot be checked until a run records a manifest.
            if self.primary == "auto":
                raise RuntimeError(
                    f"No {manifest.FILENAME} in {self.model_dir} records a champion and api.primary_model is auto; "
                    f"run `python -m prism.training.train --track {self.schema.name} --model all` or name a model."
                )
            print(
                f"[inference] no {manifest.FILENAME} in {self.model_dir}: preprocessing consistency "
                "unverified; run `python -m prism.training.train --model all` to record one"
            )
            threshold_file = self.model_dir / "thresholds.json"
            self.thresholds = (
                json.loads(threshold_file.read_text(encoding="utf-8")) if threshold_file.exists() else {}
            )
        untuned = sorted(name for name in self.models if name not in self.thresholds)
        if untuned:
            raise RuntimeError(
                f"No tuned threshold for {', '.join(untuned)} in {self.model_dir}; a model served at a "
                "default cutoff alerts at the wrong rate. Run `python -m prism.training.train "
                f"--track {self.schema.name} --model all`."
            )

        # Without a manifest nothing vouches for a champion; serve what loaded
        # and say so. With one, the checks above have already refused.
        if self.primary not in self.models:
            fallback = next(iter(self.models))
            print(f"[inference] primary model {self.primary!r} absent; serving {fallback!r} as primary")
            self.primary = fallback

        primary_model = self.models[self.primary]
        try:
            self.explainer = ShapEngine.for_model(
                primary_model, self.feature_order, schema=self.schema
            )
        except (NotImplementedError, ValueError) as exc:
            # A missing explainer degrades the response; it should not stop the
            # service from scoring transactions.
            print(f"[inference] SHAP unavailable for {self.primary}: {exc}")
            self.explainer = None

        # Reference material for the plain-English explanation. A model
        # directory trained before it existed still serves; the narrative
        # falls back to attribution only rather than the service refusing
        # to start.
        loaded = artifacts.load(self.model_dir)
        if loaded is None:
            print(
                "[inference] explanation reference absent: run "
                f"`python -m prism.explain.artifacts --track {self.schema.name}`"
            )
        else:
            self.reference, self.cases = loaded

        self._warm_up()
        return self

    def _synthetic_row(self) -> dict[str, float | str]:
        """A row of zeros and first categories, for the warm-up only."""
        row: dict[str, float | str] = {}
        for name in self.schema.input_columns:
            if name in self.schema.categorical_columns:
                position = list(self.schema.categorical_columns).index(name)
                row[name] = str(self.encoder.categories_[position][0])
            elif name == self.schema.card_column:
                row[name] = "warm-up"
            else:
                row[name] = 0.0
        return row

    def _warm_up(self) -> None:
        """Score and explain one synthetic row before the first real request.

        Measured on the real models, the first flagged transaction after startup
        took 1.6 s end to end against about 50 ms once warm, because the SHAP
        explainer and the case index build their working buffers on first use.
        Paying that once here keeps it out of the first alert an analyst sees.
        The warm-up row is discarded from the latency the service reports.
        """
        result = self.score(self._synthetic_row(), history=[])
        self.explain(result["frame"], raw=result["features"])
        self._latencies.clear()
        self.warmed_up = True

    def threshold_for(self, name: str) -> float:
        return float(self.thresholds[name])

    def derive(
        self, features: dict[str, float | str], history: History | None
    ) -> dict[str, float | str]:
        """The inputs plus what the service computes itself (design D3).

        The velocity features come from the card's earlier transactions, which
        the caller reads from the audit log; a card with no history is a new
        card, not an error.
        """
        if not self.schema.derived_columns:
            return dict(features)
        derived = velocity.velocity_row(
            history or [],
            float(features[self.schema.time_column]),
            str(features[velocity.MCC_COLUMN]),
            str(features[velocity.CHANNEL_COLUMN]),
        )
        if ratios.LIMIT_COLUMN in features and ratios.BALANCE_COLUMN in features:
            derived.update(ratios.limit_ratios(
                float(features[self.schema.amount_column]),
                float(features[ratios.LIMIT_COLUMN]),
                float(features[ratios.BALANCE_COLUMN]),
            ))
        return {**features, **derived}

    def _to_frame(self, features: dict[str, float | str]) -> pd.DataFrame:
        # Delegates to the training-time preprocessing helper so serving and
        # training can never drift apart.
        return transform_features(
            features,
            self.scaler,
            self.feature_order,
            self.scale_columns,
            encoder=self.encoder,
            categorical_columns=self.schema.categorical_columns,
        )

    def _score_one(self, name: str, frame: pd.DataFrame) -> dict:
        risk = float(self.models[name].risk_score(frame)[0])
        threshold = self.threshold_for(name)
        return {"model_name": name, "risk_score": risk, "threshold": threshold, "flagged": risk >= threshold}

    def score(
        self,
        features: dict[str, float | str],
        history: History | None = None,
        models: Sequence[str] | None = None,
    ) -> dict:
        """Score one transaction with the champion (the default) or the named models.

        The reported latency is engine time: the derivation, the transform and
        the models scored here. Challengers run off the request path through
        ``score_challengers`` on the frame this call returns. The audit-log
        lookup that supplies ``history`` runs in the endpoint before this clock
        starts; on SQLite it costs about a millisecond.
        """
        started = time.perf_counter()
        full = self.derive(features, history)
        frame = self._to_frame(full)
        names = list(models) if models is not None else [self.primary]
        scores = [self._score_one(name, frame) for name in names]
        primary = next((s for s in scores if s["model_name"] == self.primary), scores[0])
        latency_ms = (time.perf_counter() - started) * 1000
        self._latencies.append(latency_ms)

        return {
            "scores": scores,
            "primary_model": self.primary,
            "risk_score": primary["risk_score"],
            "flagged": primary["flagged"],
            "latency_ms": latency_ms,
            "frame": frame,
            "features": full,
        }

    def score_challengers(self, frame: pd.DataFrame) -> list[dict]:
        """Every model but the champion, on the frame the champion scored."""
        return [self._score_one(name, frame) for name in self.models if name != self.primary]

    def explain(
        self,
        frame: pd.DataFrame,
        raw: dict[str, float | str] | None = None,
        top_k: int = 8,
    ) -> dict | None:
        """SHAP attribution for one scaled row, plus the measured facts and the
        plain-English sentence when the reference material is loaded and the
        raw (unscaled, derived) features are supplied."""
        if self.explainer is None:
            return None
        explanation: Explanation = self.explainer.explain(frame, raw=raw)
        payload = explanation.as_dict(k=top_k)
        payload["model_name"] = payload.pop("model")
        payload["facts"] = None

        if self.reference is not None and self.cases is not None and raw is not None:
            facts = {**self.reference.describe(raw), **self.cases.query(frame)}
            found = self.clearing(frame, raw)
            facts["clearing_amount"] = found["amount"] if found else None
            facts["clearing_is_monotone"] = found["monotone"] if found else None
            payload["facts"] = facts
            payload["narrative"] = plain_english(
                explanation,
                facts,
                driver=driver_sentence(explanation, self.reference, raw, self.schema),
            )
        return payload

    def clearing(self, frame: pd.DataFrame, raw: dict[str, float | str]) -> dict | None:
        """The highest lower amount at which the champion would have cleared
        this transaction with everything else unchanged, and whether every
        amount below it clears too. None if no lower amount clears.

        Amount is the one named feature an analyst can act on. The amount and
        every feature derived from it are recomputed for each candidate and
        scaled as the service scales them, so the swept rows are exactly what
        the model would have seen; a tree model is not monotone in the amount,
        so the sentence may only claim what the sweep found."""
        amount_column = self.schema.amount_column
        amount = float(raw[amount_column])
        if amount <= 0.0:
            return None
        steps = int(config.get("explain.counterfactual_steps", 200))
        candidates = np.linspace(amount, 0.0, steps)
        # Only the columns that follow the amount change along the sweep: the
        # amount itself and its share of the limit. Each is recomputed from the
        # candidate and scaled with the fitted scaler, exactly as the service
        # transforms it; every other column is the scored row's own.
        dependent: dict[str, np.ndarray] = {amount_column: candidates}
        if ratios.LIMIT_COLUMN in raw and ratios.BALANCE_COLUMN in raw and "amount_to_limit" in self.feature_order:
            dependent["amount_to_limit"] = np.array([
                ratios.limit_ratios(float(c), float(raw[ratios.LIMIT_COLUMN]), float(raw[ratios.BALANCE_COLUMN]))["amount_to_limit"]
                for c in candidates
            ])
        sweep = pd.concat([frame] * steps, ignore_index=True)
        for column, values in dependent.items():
            if column in self.scale_columns:
                position = self.scale_columns.index(column)
                values = (values - self.scaler.mean_[position]) / self.scaler.scale_[position]
            sweep[column] = values
        threshold = self.threshold_for(self.primary)
        scores = self.models[self.primary].risk_score(sweep)
        cleared = np.flatnonzero(scores < threshold)
        if not len(cleared):
            return None
        first = int(cleared[0])
        return {"amount": float(candidates[first]), "monotone": bool(np.all(scores[first:] < threshold))}

    def clearing_amount(self, frame: pd.DataFrame, raw: dict[str, float | str]) -> float | None:
        found = self.clearing(frame, raw)
        return found["amount"] if found else None

    @property
    def mean_latency_ms(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
