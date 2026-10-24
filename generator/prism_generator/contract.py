"""The detector contract (spec/19_detector_contract.yaml): the one machine-readable description
of what a detector receives, what it is told afterwards, and what the export views carry.

The column lists of the export and the field lists of the stream are read from here, so a
column cannot change without the contract changing; `violations` says whether a payload
conforms, field by field, so a detector can refuse what it does not understand.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from .spec_loader import _find_spec_dir

CONTRACT_FILE = "19_detector_contract.yaml"

_PYTHON_TYPES: Dict[str, Tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool, int),
}


@dataclass(frozen=True)
class ContractField:
    name: str
    type: str
    description: str
    domain: Tuple[str, ...] = ()
    minimum: float | None = None


@dataclass(frozen=True)
class DetectorContract:
    version: int
    feeds: Dict[str, Tuple[ContractField, ...]]
    export_views: Dict[str, Tuple[str, ...]]
    raw: Dict[str, Any] = field(default_factory=dict, compare=False)

    def fields(self, feed: str) -> Tuple[ContractField, ...]:
        if feed not in self.feeds:
            raise KeyError(f"The contract has no feed {feed!r}; it has {sorted(self.feeds)}")
        return self.feeds[feed]

    def field_names(self, feed: str) -> List[str]:
        return [f.name for f in self.fields(feed)]

    def export_view(self, name: str) -> Tuple[str, ...]:
        if name not in self.export_views:
            raise KeyError(f"The contract has no export view {name!r}; it has {sorted(self.export_views)}")
        return self.export_views[name]

    def violations(self, feed: str, payload: Dict[str, Any]) -> List[str]:
        """Every way a payload fails the feed: a missing or extra field, a value of the
        wrong type, a value outside the field's domain or below its minimum."""
        found: List[str] = []
        expected = self.fields(feed)
        names = {f.name for f in expected}
        for extra in sorted(set(payload) - names):
            found.append(f"{extra}: not a field of {feed}")
        for spec in expected:
            if spec.name not in payload:
                found.append(f"{spec.name}: missing")
                continue
            value = payload[spec.name]
            if value is None:
                found.append(f"{spec.name}: null")
                continue
            if isinstance(value, bool) and spec.type != "boolean":
                found.append(f"{spec.name}: {value!r} is a boolean, not {spec.type}")
                continue
            if not isinstance(value, _PYTHON_TYPES[spec.type]):
                found.append(f"{spec.name}: {value!r} is not {spec.type}")
                continue
            if spec.domain and str(value) not in spec.domain:
                found.append(f"{spec.name}: {value!r} is outside the domain {list(spec.domain)}")
            if spec.minimum is not None and float(value) < spec.minimum:
                found.append(f"{spec.name}: {value!r} is below the minimum {spec.minimum}")
        return found


def _parse_fields(entries: Sequence[Dict[str, Any]]) -> Tuple[ContractField, ...]:
    return tuple(
        ContractField(
            name=str(e["name"]),
            type=str(e["type"]),
            description=str(e.get("description", "")),
            domain=tuple(str(v) for v in e.get("domain", [])),
            minimum=float(e["minimum"]) if "minimum" in e else None,
        )
        for e in entries
    )


@functools.lru_cache(maxsize=1)
def load_contract() -> DetectorContract:
    """The contract, parsed once from the spec directory."""
    raw = yaml.safe_load((_find_spec_dir() / CONTRACT_FILE).read_text(encoding="utf-8"))
    feeds = {str(name): _parse_fields(body["fields"]) for name, body in raw["feeds"].items()}
    views = {str(name): tuple(str(c) for c in body["columns"]) for name, body in raw["export_views"].items()}
    return DetectorContract(version=int(raw["contract_version"]), feeds=feeds, export_views=views, raw=raw)
