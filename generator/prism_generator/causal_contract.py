"""The causal-explanation contract: the eleven explained columns and the canonical
intervention supports, each declared once and read from `spec/21`.

Two harnesses explain the same eleven columns and score them against the same ground
truths, so those eleven columns are declared here and nowhere else:

- `benchmark.py` (the XAI harness): Track 1 forensic intervention recovery and the
  concordance of the post-hoc explainer with the closed-form attributions.
- `benchmark_reporter.py` (the four-pillar runner): the Pillar-4 causal scorecard.

Column names are the record/lever vocabulary used by `canonical_interventions` and by
the HeuristicBankScorer input vector (`causal_scm.py`), so a playbook's declared levers
intersect the modelled columns **by name**. That is the whole point: the three copies
this module replaces had drifted into three vocabularies, and one of the copies
(`benchmark.py`) contradicted `spec/21` for seven of the eight playbooks they shared.

Sources
-------
- `spec/21_prism_causal_evaluation.yaml` -> `explained_feature_contract` owns the
  columns, their sources and their ground-truth keys; `canonical_interventions` owns
  the per-playbook physical levers.
- `causal_scm.py` -> `HeuristicBankScorer` emits `analytical_shapley_probability` and
  `analytical_shapley_log_odds`, and owns the ground-truth key names.
- `contract.py` / `spec/19` owns the authorisation feed the record is drawn from.

Known limits, stated so a scorecard is read correctly
-----------------------------------------------------
- Only the eleven modelled columns can be scored. A lever a playbook declares outside
  this contract (`channel_type`, `pos_entry_mode`, `asn_type`, `vaai_score`,
  `available_balance`, `credit_limit`, `eci`, `trans_status_3ds`, `mcc`) is reported by
  `load_intervention_projection` as unmodelled: an explainer that never sees a lever
  cannot be credited with recovering it.
- `amount` is modelled as the raw authorised ticket while its ground-truth key is
  `amount_to_mean_ratio_30d`, the 30-day ratio the scorer consumes: a documented proxy.
- The five authentication deviations (`emv_arqc_verified`, `emv_pin_verified`,
  `emv_tvr_clean`, `three_ds_authenticated`, `three_ds_attempted`) are read verbatim from
  the record. `rails.RailSwitch.authenticate()` decides them before the issuer's
  authorisation decision and the engine writes them onto the record before `evaluate()`
  runs, so the causal ground truth and the explained model consume one label-free outcome
  (2026-10-16; before that the two columns read here were never emitted and the scorer's
  fallback had to infer them - see docs/PROJECT_CONTEXT.md 7.1 item 2).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Tuple

import yaml

from .spec_loader import _find_spec_dir

SPEC_FILE = "21_prism_causal_evaluation.yaml"

GROUND_TRUTH_FIELDS: Mapping[str, str] = {
    "probability": "analytical_shapley_probability",
    "log_odds": "analytical_shapley_log_odds",
}

# Record fields the contract reads that the simulator does not emit yet. A *new* missing
# field fails `tests/test_causal_contract.py`, so this mapping is empty and stays empty:
# any future gap must be named here and given a measurement, as these two were.
#
# History, kept because the empty mapping looks like an oversight otherwise. Until
# 2026-10-16 this listed `emv_arqc_verified` and `three_ds_authenticated`: neither key was
# written to any record, so both columns were constant 0.0 in every explained model while
# the closed-form target attributed to them (4,323 of 7,500 US records carried a negative
# `emv_arqc_verified` component, 2,555 a `three_ds_authenticated` one). The rails now decide
# authentication before the authorisation step and the engine attaches all five deviations
# to the record before `evaluate()` reads them, so both columns are live. The change moved
# published metrics and is recorded in docs/PROJECT_CONTEXT.md 7.1.
KNOWN_UNEMITTED_FIELDS: Mapping[str, str] = {}


@dataclass(frozen=True)
class CausalFeature:
    """One explained column and the ground-truth key it is scored against."""

    name: str
    kind: str
    ground_truth_key: str
    source: Mapping[str, str]
    note: str = ""

    @property
    def modelled(self) -> bool:
        return True


def _avs_mismatch(record: Mapping[str, Any]) -> float:
    """1 when the address verification says no match or unavailable (spec/19 avs_match_code)."""
    return 1.0 if str(record.get("avs_match_code", "Y")) in ("N", "U") else 0.0


def _billing_shipping_mismatch(record: Mapping[str, Any]) -> float:
    return 1.0 if int(record.get("billing_shipping_match", 1)) == 0 else 0.0


def _cvv_mismatch(record: Mapping[str, Any]) -> float:
    """1 when the card verification value did not match; mismatch polarity, as the scorer consumes it."""
    return 1.0 if int(record.get("cvv_match_flag", 1)) == 0 else 0.0


# The closed rule vocabulary `explained_feature_contract` may name. A rule that is not
# here is a spec typo and raises rather than silently scoring a constant column.
_RULES: Mapping[str, Callable[[Mapping[str, Any]], float]] = {
    "avs_mismatch": _avs_mismatch,
    "billing_shipping_mismatch": _billing_shipping_mismatch,
    "cvv_mismatch": _cvv_mismatch,
}

_CASTS: Mapping[str, Callable[[Any], float]] = {
    "bool01": lambda value: 1.0 if value else 0.0,
}


def _extractor(feature: CausalFeature) -> Callable[[Mapping[str, Any]], float]:
    source = feature.source
    if "rule" in source:
        rule_id = str(source["rule"])
        if rule_id not in _RULES:
            raise KeyError(
                f"explained_feature_contract.{feature.name} names rule {rule_id!r}, "
                f"which is not implemented; implemented rules are {sorted(_RULES)}"
            )
        return _RULES[rule_id]
    if "field" in source:
        field_name = str(source["field"])
        cast = source.get("cast")
        if cast is not None:
            if str(cast) not in _CASTS:
                raise KeyError(
                    f"explained_feature_contract.{feature.name} names cast {cast!r}, "
                    f"which is not implemented; implemented casts are {sorted(_CASTS)}"
                )
            cast_fn = _CASTS[str(cast)]
            return lambda record: cast_fn(record.get(field_name))
        return lambda record: float(record.get(field_name, 0.0) or 0.0)
    raise KeyError(
        f"explained_feature_contract.{feature.name} declares no source; "
        "use {field: ...} or {rule: ...}"
    )


@functools.lru_cache(maxsize=1)
def load_causal_contract() -> Tuple[CausalFeature, ...]:
    """The eleven explained columns, parsed once from spec/21."""
    raw = yaml.safe_load((_find_spec_dir() / SPEC_FILE).read_text(encoding="utf-8"))
    entries = raw["explained_feature_contract"]["features"]
    features = tuple(
        CausalFeature(
            name=str(entry["name"]),
            kind=str(entry["kind"]),
            ground_truth_key=str(entry["ground_truth_key"]),
            source={str(k): str(v) for k, v in dict(entry["source"]).items()},
            note=str(entry.get("note", "")),
        )
        for entry in entries
    )
    names = [f.name for f in features]
    if len(names) != len(set(names)):
        raise ValueError(f"explained_feature_contract has duplicate column names: {names}")
    return features


@functools.lru_cache(maxsize=1)
def _extractors() -> Mapping[str, Callable[[Mapping[str, Any]], float]]:
    return {feature.name: _extractor(feature) for feature in load_causal_contract()}


def feature_names() -> Tuple[str, ...]:
    """The explained columns, in contract order."""
    return tuple(feature.name for feature in load_causal_contract())


def ground_truth_keys() -> Mapping[str, str]:
    """Column name -> key in `analytical_shapley_probability` / `_log_odds`."""
    return {feature.name: feature.ground_truth_key for feature in load_causal_contract()}


def feature_vector(record: Mapping[str, Any]) -> List[float]:
    """The eleven explained columns for one record, in contract order."""
    extractors = _extractors()
    return [extractors[name](record) for name in feature_names()]


def ground_truth_attributions(
    record: Mapping[str, Any],
    scale: str = "auto",
) -> List[float]:
    """The record's closed-form attributions for the eleven columns, in contract order.

    `scale` is "probability" (128-point Gauss-Legendre), "log_odds" (exact Owen
    multilinear extension) or "auto" (log-odds when the record carries them, else
    probability).
    """
    if scale == "auto":
        chosen = "__missing__"
        for candidate in ("log_odds", "probability"):
            if GROUND_TRUTH_FIELDS[candidate] in record:
                chosen = candidate
                break
        if chosen == "__missing__":
            chosen = "probability"
    elif scale in GROUND_TRUTH_FIELDS:
        chosen = scale
    else:
        raise ValueError(f"unknown ground-truth scale {scale!r}; expected one of {sorted(GROUND_TRUTH_FIELDS)} or 'auto'")
    attributions = record.get(GROUND_TRUTH_FIELDS[chosen]) or {}
    return [float(attributions.get(feature.ground_truth_key, 0.0)) for feature in load_causal_contract()]


@dataclass(frozen=True)
class InterventionProjection:
    """spec/21's declared physical levers, projected onto the modelled columns."""

    declared: Mapping[str, Tuple[str, ...]]
    support: Mapping[str, frozenset[str]]
    unmodelled: Mapping[str, Tuple[str, ...]]

    def declared_levers(self, playbook: str) -> Tuple[str, ...]:
        return self.declared.get(playbook, ())

    def support_set(self, playbook: str) -> frozenset[str]:
        """The modelled levers a playbook physically intervenes on (empty when undeclared)."""
        return self.support.get(playbook, frozenset())

    def unmodelled_levers(self, playbook: str) -> Tuple[str, ...]:
        """Levers the playbook declares that the explained model never receives."""
        return self.unmodelled.get(playbook, ())

    def all_unmodelled_levers(self) -> Tuple[str, ...]:
        return tuple(sorted({lever for levers in self.unmodelled.values() for lever in levers}))


@functools.lru_cache(maxsize=1)
def load_intervention_projection() -> InterventionProjection:
    """spec/21's `canonical_interventions`, projected onto the explained columns."""
    raw = yaml.safe_load((_find_spec_dir() / SPEC_FILE).read_text(encoding="utf-8"))
    modelled = set(feature_names())
    declared: Dict[str, Tuple[str, ...]] = {}
    support: Dict[str, frozenset[str]] = {}
    unmodelled: Dict[str, Tuple[str, ...]] = {}
    for playbook, body in raw["canonical_interventions"].items():
        levers = tuple(str(lever) for lever in body["intervened_features"])
        declared[str(playbook)] = levers
        support[str(playbook)] = frozenset(lever for lever in levers if lever in modelled)
        unmodelled[str(playbook)] = tuple(lever for lever in levers if lever not in modelled)
    return InterventionProjection(declared=declared, support=support, unmodelled=unmodelled)


def intervention_support(playbook: str) -> frozenset[str]:
    """The modelled physical levers for one playbook tag (empty when spec/21 declares none)."""
    return load_intervention_projection().support_set(playbook)


def declared_playbooks() -> Tuple[str, ...]:
    """Every playbook tag spec/21 declares an intervention for."""
    return tuple(sorted(load_intervention_projection().declared))
