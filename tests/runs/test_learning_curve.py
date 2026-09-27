"""PR-046: the nested-prefix learning curve and D43's "history-limited" verdict.

Program v0.3 D43: "a learning curve on nested-prefix training windows (1/4, 1/2,
3/4, full ...); 'history-limited' fires when the OOS metric still improves from
3/4 to full by more than a tolerance" — the tolerance is the operator-signed
``m9_learning_curve_tolerance_rel``. Expected values are hand-computed.
"""

from __future__ import annotations

import polars as pl
import pytest

from rux_ml.data.splits import temporal_train_val_test_split, train_prefix
from rux_ml.runs.learning_curve import LearningCurveError, learning_curve_report

FRACS = [0.25, 0.5, 0.75, 1.0]


def test_train_prefix_keeps_the_first_fraction_of_the_unique_stamps() -> None:
    """Train stamps 0..11 (12 unique, two rows each): f = 0.5 keeps ceil(6) = 6 stamps
    0..5; f = 0.25 keeps 3; f = 1.0 keeps all."""
    df = pl.DataFrame({"t": [t for t in range(12) for _ in range(2)], "x": [0.0] * 24})
    assert train_prefix(df, time_column="t", frac=0.5)["t"].unique().sort().to_list() == list(
        range(6)
    )
    assert train_prefix(df, time_column="t", frac=0.25).height == 6
    assert train_prefix(df, time_column="t", frac=1.0).equals(df)
    with pytest.raises(ValueError, match="time_column"):
        train_prefix(df, time_column="missing", frac=0.5)


def test_prefixes_nest_and_leave_val_and_test_alone() -> None:
    df = pl.DataFrame({"t": list(range(40)), "x": [0.0] * 40})
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    full = temporal_train_val_test_split(df, time_column="t", ratios=ratios, embargo=2)
    sets = [set(train_prefix(full["train"], time_column="t", frac=f)["t"]) for f in FRACS]
    assert sets[0] < sets[1] < sets[2] < sets[3] == set(full["train"]["t"])


def _points(values: list[float], n_test: int = 100) -> list[dict[str, object]]:
    return [
        {"study": f"s{i}", "trial": 0, "train_prefix_frac": f, "split_kind": "time_ordered",
         "oos": {"n_rows": n_test, "score": v, "honesty": {"mae_bp": v}}}
        for i, (f, v) in enumerate(zip(FRACS, values, strict=True))
    ]  # fmt: skip


def _report(points: list[dict[str, object]], **kw: object) -> dict[str, object]:
    args: dict[str, object] = {
        "metric_path": "score",
        "direction": "minimize",
        "tolerance_rel": 0.05,
    }
    args.update(kw)
    return learning_curve_report(points, fractions=FRACS, **args)  # type: ignore[arg-type]


def test_history_limited_fires_above_the_tolerance() -> None:
    """MAE 3.0, 2.4, 2.0, 1.8: (2.0 - 1.8) / 2.0 = 0.10 > 0.05 -> fires."""
    r = _report(_points([3.0, 2.4, 2.0, 1.8]))
    assert r["curve"] == [
        {"train_prefix_frac": f, "value": v, "study": f"s{i}", "trial": 0}
        for i, (f, v) in enumerate(zip(FRACS, [3.0, 2.4, 2.0, 1.8], strict=True))
    ]
    assert r["improvement_rel_last_step"] == pytest.approx(0.10)
    assert r["history_limited"] is True


def test_history_limited_does_not_fire_at_or_below_the_tolerance() -> None:
    """(2.0 - 1.95) / 2.0 = 0.025 <= 0.05; a worse full fit is a negative improvement."""
    assert _report(_points([3.0, 2.4, 2.0, 1.95]))["history_limited"] is False
    r = _report(_points([3.0, 2.4, 2.0, 2.2]))
    assert r["improvement_rel_last_step"] == pytest.approx(-0.10)
    assert r["history_limited"] is False


def test_a_maximized_metric_and_a_nested_path() -> None:
    """A fraction to maximize: 0.80 -> 0.90 is (0.90 - 0.80) / 0.80 = 0.125 -> fires."""
    r = _report(_points([0.5, 0.7, 0.8, 0.9]), metric_path="honesty.mae_bp", direction="maximize")
    assert r["improvement_rel_last_step"] == pytest.approx(0.125)
    assert r["history_limited"] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p[:3], "fractions"),
        (lambda p: [*p, p[0]], "fractions"),
        (lambda p: [*p[:3], {**p[3], "oos": {**p[3]["oos"], "n_rows": 99}}], "test partition"),
        (lambda p: [*p[:3], {**p[3], "split_kind": "random"}], "regime"),
        (lambda p: [*p[:3], {**p[3], "oos": {"n_rows": 100}}], "score"),
    ],
)
def test_an_incomparable_curve_is_refused(mutate: object, match: str) -> None:
    with pytest.raises(LearningCurveError, match=match):
        _report(mutate(_points([3.0, 2.4, 2.0, 1.8])))  # type: ignore[operator]
