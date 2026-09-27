"""The nested-prefix learning curve and D43's "history-limited" verdict (PR-046).

Program v0.3 D43: per target, fits on nested-prefix training windows (1/4, 1/2,
3/4, full); "history-limited" fires when the OOS metric still improves from the
second-to-last prefix to the full window by more than a tolerance — the
operator-signed ``m9_learning_curve_tolerance_rel``, passed in.

A curve is only a curve when its points are comparable: one regime, one OOS
partition (same row count), each configured fraction exactly once. Anything else
is refused with :class:`LearningCurveError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# A curve needs a last step: the full window and the prefix before it.
_MIN_POINTS = 2


class LearningCurveError(ValueError):
    """An incomparable or incomplete curve (a ``ValueError``: exit 2)."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"learning curve: {detail} — refused")


def _dig(record: dict[str, Any], path: str) -> float:
    node: Any = record
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            msg = f"the oos record has no {path!r}"
            raise LearningCurveError(msg)
        node = node[key]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(node, bool) or not isinstance(node, int | float):
        msg = f"oos {path!r} = {node!r} is not a number"
        raise LearningCurveError(msg)
    return float(node)


def learning_curve_report(
    points: Sequence[dict[str, Any]],
    *,
    fractions: Sequence[float],
    metric_path: str,
    direction: str,
    tolerance_rel: float,
) -> dict[str, Any]:
    """Order the points by prefix, check they are one curve, and apply D43.

    Each point: ``{"study", "trial", "train_prefix_frac", "split_kind", "oos"}``.
    Improvement of the last step, relative to the second-to-last value:
    ``(prev - last) / |prev|`` to minimize, ``(last - prev) / |prev|`` to maximize;
    history-limited when it exceeds ``tolerance_rel``.
    """
    wanted = sorted(fractions)
    got = sorted(float(p["train_prefix_frac"]) for p in points)
    if got != wanted:
        msg = f"prefix fractions {got} != the configured fractions {wanted}"
        raise LearningCurveError(msg)
    if len(wanted) < _MIN_POINTS:
        msg = "fewer than two fractions configured"
        raise LearningCurveError(msg)
    kinds = {p["split_kind"] for p in points}
    if len(kinds) != 1:
        msg = f"points from more than one regime {sorted(kinds)}"
        raise LearningCurveError(msg)
    sizes = {p["oos"].get("n_rows") for p in points}
    if len(sizes) != 1:
        msg = f"points scored on different test partitions (n_rows {sorted(map(str, sizes))})"
        raise LearningCurveError(msg)
    ordered = sorted(points, key=lambda p: float(p["train_prefix_frac"]))
    curve = [
        {
            "train_prefix_frac": float(p["train_prefix_frac"]),
            "value": _dig(p["oos"], metric_path),
            "study": p["study"],
            "trial": p["trial"],
        }
        for p in ordered
    ]
    prev, last = curve[-2]["value"], curve[-1]["value"]
    if prev == 0.0:
        msg = f"the second-to-last value of {metric_path!r} is 0: no relative improvement"
        raise LearningCurveError(msg)
    delta = (prev - last) if direction == "minimize" else (last - prev)
    improvement = delta / abs(prev)
    return {
        "regime": next(iter(kinds)),
        "metric": metric_path,
        "direction": direction,
        "curve": curve,
        "improvement_rel_last_step": improvement,
        "tolerance_rel": tolerance_rel,
        "history_limited": improvement > tolerance_rel,
    }
