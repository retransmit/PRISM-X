"""The detector contract as the service reads it (generator/spec/19_detector_contract.yaml).

The shipped generator defines what an authorisation request carries: each field's type and,
for the code sets of the payment rails, the values it may take. The request model checks a
request against the feed its track names, so a producer that drifts from the contract gets a
422 naming the field and the value rather than a score computed on nonsense.
"""

from __future__ import annotations

import functools
from typing import Any

# Vocabularies that grow in production: an acquirer, a gateway, a product, a persona, a
# merchant. A value the generator never emitted still scores. The code sets of the rails
# (channel, entry mode, condition code, ECI, the 3-D Secure result, AVS, the ASN type, the
# currency, the message type) are closed, and a value outside them is refused.
OPEN_VOCABULARIES = frozenset({"acquirer_bin", "gateway_provider", "product_id", "cohort_id", "merchant_id", "mid", "tid"})

_TRUE = {True, 1, "1", "true", "True", "TRUE"}
_FALSE = {False, 0, "0", "false", "False", "FALSE"}


@functools.lru_cache(maxsize=1)
def load():
    """The contract parsed by the shipped generator's own loader."""
    try:
        from prism_generator.contract import load_contract
    except ImportError as exc:  # pragma: no cover - the requirements install the generator
        raise RuntimeError("The generator's contract needs the shipped generator: pip install -e generator/") from exc
    return load_contract()


def fields(feed: str):
    return load().fields(feed)


def _is_open(spec: Any) -> bool:
    return spec.name in OPEN_VOCABULARIES or bool(getattr(spec, "open_vocabulary", False))


def violations(feed: str, supplied: dict[str, Any]) -> list[str]:
    """How the supplied values fail the fields of the feed they name: a wrong type, or a
    value outside a closed code set. Fields the request does not carry are not judged here,
    the track says what is required; open vocabularies are not judged at all."""
    found: list[str] = []
    for spec in load().fields(feed):
        if spec.name not in supplied:
            continue
        value = supplied[spec.name]
        if value is None:
            found.append(f"{spec.name} is null")
            continue
        if spec.type == "boolean":
            if isinstance(value, bool) or value in _TRUE or value in _FALSE:
                text = "True" if (value in _TRUE) else "False"
            else:
                found.append(f"{spec.name} must be true or false, not {value!r}")
                continue
        elif spec.type in ("number", "integer"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                found.append(f"{spec.name} must be a number, not {value!r}")
                continue
            if spec.type == "integer" and not number.is_integer():
                found.append(f"{spec.name} must be a whole number, not {value!r}")
                continue
            text = str(int(number)) if spec.type == "integer" else str(value)
        else:
            text = str(value)
        if spec.domain and not _is_open(spec) and text not in spec.domain:
            found.append(f"{spec.name} must be one of {list(spec.domain)}, not {value!r}")
    return found
