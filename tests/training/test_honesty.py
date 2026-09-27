"""PR-044: the signed-error honesty test (program v0.3 D41, D45 #3; gen-2's criterion).

Both columns are costs in bp (higher = worse). (i) no under-deduction: the
predicted cost >= the realized on at least ``no_underdeduct_frac_min`` of the rows;
(ii) the over-deduction ``(mean(pred) - mean(realized)) / |mean(realized)|`` at most
``overdeduct_max_rel``. Expected values are computed by hand below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rux_ml.training.honesty import signed_error_honesty


def test_hand_computed_example() -> None:
    pred = np.array([2.0, 3.0, 5.0, 1.0, 4.0])
    real = np.array([1.0, 3.0, 4.0, 2.0, np.nan])  # the last realized is undefined
    r = signed_error_honesty(pred, real, no_underdeduct_frac_min=0.75, overdeduct_max_rel=0.2)
    assert r["n"] == 4 and r["n_realized_missing"] == 1
    # pred >= real on rows 0, 1, 2 (equality counts): 3 of 4
    assert r["no_underdeduct_frac"] == 0.75
    # mean pred over the 4 rows = 11/4, mean real = 10/4 -> (2.75 - 2.5) / 2.5 = 0.1
    assert math.isclose(r["overdeduct_rel"], 0.1)
    assert math.isclose(r["mae_bp"], (1 + 0 + 1 + 1) / 4)
    assert math.isclose(r["mean_signed_error_bp"], (1 + 0 + 1 - 1) / 4)
    assert r["passes_no_underdeduct"] and r["passes_overdeduct"] and r["honest"]


def test_each_criterion_can_fail_alone() -> None:
    pred = np.array([2.0, 3.0, 5.0, 1.0])
    real = np.array([1.0, 3.0, 4.0, 2.0])
    r = signed_error_honesty(pred, real, no_underdeduct_frac_min=0.8, overdeduct_max_rel=0.2)
    assert not r["passes_no_underdeduct"] and r["passes_overdeduct"] and not r["honest"]
    r = signed_error_honesty(pred, real, no_underdeduct_frac_min=0.75, overdeduct_max_rel=0.05)
    assert r["passes_no_underdeduct"] and not r["passes_overdeduct"] and not r["honest"]


def test_a_zero_mean_realized_cost_is_not_evaluable_never_a_pass() -> None:
    r = signed_error_honesty(
        np.array([1.0, -1.0]), np.array([1.0, -1.0]), no_underdeduct_frac_min=0.5,
        overdeduct_max_rel=0.3,
    )  # fmt: skip
    assert r["overdeduct_rel"] is None
    assert r["passes_overdeduct"] is False and r["honest"] is False


def test_a_missing_prediction_or_no_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="prediction"):
        signed_error_honesty(
            np.array([1.0, np.nan]), np.array([1.0, 1.0]), no_underdeduct_frac_min=0.9,
            overdeduct_max_rel=0.3,
        )  # fmt: skip
    with pytest.raises(ValueError, match="no row"):
        signed_error_honesty(
            np.array([1.0]), np.array([np.nan]), no_underdeduct_frac_min=0.9,
            overdeduct_max_rel=0.3,
        )  # fmt: skip
