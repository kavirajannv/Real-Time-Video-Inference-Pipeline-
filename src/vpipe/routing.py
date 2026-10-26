"""Conditional routing rule: Model A result -> submit to Model B or not."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from .schema import SKIP_REASON

log = logging.getLogger(__name__)


def should_route(label: str | None, conf: float | None, route_labels: Iterable[str], threshold: float) -> tuple[bool, str]:
    """Route only when the *top-1* label is a target label and its confidence >= threshold."""
    if label is None or conf is None:
        return False, "No Model A result"
    if label in set(route_labels) and conf >= threshold:
        return True, f"top-1 '{label}' conf {conf:.2f} >= {threshold:.2f}"
    return False, SKIP_REASON


def verify_route_labels(route_labels: Iterable[str], model_names: dict[int, str]) -> tuple[list[str], list[str]]:
    """Check target labels against the classifier's real label set (never trust hardcoded indices).

    Returns (present, missing). A missing label can never be predicted, so it is logged loudly.
    """
    available = set(model_names.values())
    labels = list(dict.fromkeys(route_labels))
    present = [l for l in labels if l in available]
    missing = [l for l in labels if l not in available]
    for name in present:
        idx = next(i for i, n in model_names.items() if n == name)
        log.info("routing label %r verified -> class index %d", name, idx)
    if missing:
        log.warning(
            "routing label(s) %s do not exist in Model A's %d classes; those labels can never be "
            "predicted, so frames will only route via: %s",
            missing, len(available), present or "NOTHING (no frame will reach Model B)",
        )
    return present, missing
