"""Tests for the pruner factory (sub-decision A1)."""

from __future__ import annotations

import pytest
from optuna.pruners import (
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    SuccessiveHalvingPruner,
    WilcoxonPruner,
)

from rux_ml.config import TuningConfig
from rux_ml.tuning import make_pruner


@pytest.mark.parametrize(
    ("kind", "expected_cls"),
    [
        ("wilcoxon", WilcoxonPruner),
        ("median", MedianPruner),
        ("hyperband", HyperbandPruner),
        ("successive_halving", SuccessiveHalvingPruner),
        ("none", NopPruner),
    ],
)
def test_make_pruner_dispatches_by_kind(kind: str, expected_cls: type) -> None:
    cfg = TuningConfig(pruner=kind)  # type: ignore[arg-type]
    pruner = make_pruner(cfg)
    assert isinstance(pruner, expected_cls)


def test_make_pruner_default_is_wilcoxon() -> None:
    """PR-007 sub-decision A1: WilcoxonPruner default for the CV-mean regime."""
    cfg = TuningConfig()
    pruner = make_pruner(cfg)
    assert isinstance(pruner, WilcoxonPruner)
