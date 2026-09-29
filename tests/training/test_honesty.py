"""PR-044: the signed-error honesty test (program v0.3 D41, D45 #3; gen-2's criterion).

Both columns are costs in bp (higher = worse). (i) no under-deduction: the
predicted cost >= the realized on at least ``no_underdeduct_frac_min`` of the rows;
(ii) the over-deduction ``(mean(pred) - mean(realized)) / |mean(realized)|`` at most
``overdeduct_max_rel``. Expected values are computed by hand below.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from rux_ml.training.honesty import CalibrationBias, signed_error_honesty


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


# ---------- PR-058 (program PR-027 A7): both forms reported, the verdict's form named ----------
#
# A7 (program PR-027 :690-691, operator-approved 2026-09-28): "Markout is tested as the EV
# deducts it, max(0, m) < r". The record carries the signed form (PR-044: the prediction as
# fitted) and the clipped form max(0, pred) side by side; ``verdict_form`` names the form
# whose values PR-044's verdict keys carry. Expected values are computed by hand.

P5 = np.array([-2.0, 1.0, 3.0, -1.0, 0.5])
R5 = np.array([-1.0, 1.0, 2.0, 0.5, 1.0])
VERDICT_KEYS = (
    "no_underdeduct_frac", "overdeduct_rel", "passes_no_underdeduct", "passes_overdeduct", "honest",
)  # fmt: skip


def test_both_forms_hand_computed_and_the_verdict_follows_the_named_form() -> None:
    r = signed_error_honesty(P5, R5, no_underdeduct_frac_min=0.6, overdeduct_max_rel=0.3)
    s, c = r["forms"]["signed"], r["forms"]["clipped_at_zero"]
    # signed: p >= r on rows 1, 2 -> 2 / 5
    # clipped: max(0, p) = [0, 1, 3, 0, 0.5] >= r on rows 0, 1, 2 -> 3 / 5
    assert s["no_underdeduct_frac"] == 0.4 and c["no_underdeduct_frac"] == 0.6
    # mean r = 3.5 / 5 = 0.7; mean p = 1.5 / 5 = 0.3; mean max(0, p) = 4.5 / 5 = 0.9
    assert math.isclose(r["mean_realized_bp"], 0.7)
    assert math.isclose(s["mean_pred_bp"], 0.3) and math.isclose(c["mean_pred_bp"], 0.9)
    assert math.isclose(s["overdeduct_rel"], (0.3 - 0.7) / 0.7)  # -0.571
    assert math.isclose(c["overdeduct_rel"], (0.9 - 0.7) / 0.7)  # +0.286
    assert (s["passes_no_underdeduct"], s["passes_overdeduct"], s["honest"]) == (False, True, False)
    assert (c["passes_no_underdeduct"], c["passes_overdeduct"], c["honest"]) == (True, True, True)
    # MAE and the mean signed error stay the fitted prediction's: |e| = 1, 0, 1, 1.5, 0.5
    assert math.isclose(r["mae_bp"], 4.0 / 5) and math.isclose(r["mean_signed_error_bp"], -2.0 / 5)
    # the default verdict is PR-044's signed form, under PR-044's keys: nothing moved
    assert r["verdict_form"] == "signed"
    assert {k: r[k] for k in VERDICT_KEYS} == {k: s[k] for k in VERDICT_KEYS}
    rc = signed_error_honesty(
        P5, R5, no_underdeduct_frac_min=0.6, overdeduct_max_rel=0.3, verdict_form="clipped_at_zero"
    )
    assert rc["verdict_form"] == "clipped_at_zero" and rc["forms"] == r["forms"]
    assert {k: rc[k] for k in VERDICT_KEYS} == {k: c[k] for k in VERDICT_KEYS}
    assert rc["honest"] is True and r["honest"] is False


def test_an_all_negative_prediction_clips_to_zero() -> None:
    pred = np.array([-1.0, -2.0, -3.0, -0.5])
    real = np.array([0.5, -1.0, 2.0, -4.0])
    r = signed_error_honesty(
        pred, real, no_underdeduct_frac_min=0.5, overdeduct_max_rel=0.3,
        verdict_form="clipped_at_zero",
    )  # fmt: skip
    s, c = r["forms"]["signed"], r["forms"]["clipped_at_zero"]
    # signed: only -0.5 >= -4 -> 1 / 4; clipped: every prediction is 0, and 0 >= r on rows 1, 3
    assert s["no_underdeduct_frac"] == 0.25 and c["no_underdeduct_frac"] == 0.5
    # mean r = -2.5 / 4 = -0.625 (favourable on average); mean p = -6.5 / 4; mean max(0, p) = 0
    assert c["mean_pred_bp"] == 0.0
    assert math.isclose(s["overdeduct_rel"], (-1.625 + 0.625) / 0.625)  # -1.6
    assert math.isclose(c["overdeduct_rel"], (0.0 + 0.625) / 0.625)  # +1.0, over the 0.3 bound
    assert c["passes_no_underdeduct"] and not c["passes_overdeduct"] and not r["honest"]


def test_a_zero_mean_realized_cost_is_not_evaluable_in_either_form() -> None:
    r = signed_error_honesty(
        np.array([1.0, -1.0]), np.array([1.0, -1.0]), no_underdeduct_frac_min=0.5,
        overdeduct_max_rel=0.3, verdict_form="clipped_at_zero",
    )  # fmt: skip
    for form in r["forms"].values():
        assert form["overdeduct_rel"] is None
        assert form["passes_overdeduct"] is False and form["honest"] is False
    # max(0, p) = [1, 0] >= [1, -1] on both rows
    assert r["forms"]["clipped_at_zero"]["no_underdeduct_frac"] == 1.0
    assert r["overdeduct_rel"] is None and r["honest"] is False


def test_an_unknown_verdict_form_is_refused() -> None:
    with pytest.raises(ValueError, match="verdict_form"):
        signed_error_honesty(
            P5, R5, no_underdeduct_frac_min=0.6, overdeduct_max_rel=0.3, verdict_form="clipped"
        )


# ---------- PR-058 (program PR-027 A7): the signed calibration bias, per bucket ----------
#
# A7: "fill is gated on Brier, with a signed calibration bias per (p, Q, h) bucket reported".
# The bias is mean(pred) - mean(realized) over the rows with a realized value, overall and per
# bucket (each distinct value tuple of the bucket columns), with the bucket of the largest
# |bias|; accumulated batch by batch. Reported, never gated.

BUCKETS = ["p_bp", "q_usd", "h_ms"]


def _keys(p_bp: list[int]) -> pl.DataFrame:
    n = len(p_bp)
    return pl.DataFrame({"p_bp": p_bp, "q_usd": [1000] * n, "h_ms": [60_000] * n})


def _close(a: dict[str, object], b: dict[str, object]) -> bool:
    return a.keys() == b.keys() and all(
        math.isclose(x, b[k], rel_tol=1e-12, abs_tol=1e-15)  # type: ignore[arg-type]
        if isinstance(x, float) else x == b[k]
        for k, x in a.items()
    )  # fmt: skip


def test_calibration_bias_hand_computed_over_two_batches() -> None:
    # bucket p 10: pred 0.8, 0.6 vs realized 1.0, 1.0 -> 0.7 - 1.0 = -0.3
    # bucket p 20: pred 0.5, 0.4, 0.3 vs realized 0.5, 0.0, 0.1 -> 0.4 - 0.2 = +0.2
    # overall: 2.6 / 5 - 2.6 / 5 = 0 -- calibrated on the whole, not per bucket
    acc = CalibrationBias(BUCKETS)
    acc.add(_keys([10, 20, 10]), np.array([0.8, 0.5, 0.6]), np.array([1.0, 0.5, 1.0]))
    acc.add(_keys([20, 20]), np.array([0.4, 0.3]), np.array([0.0, 0.1]))
    rec = acc.record()
    assert rec["n"] == 5 and rec["n_realized_missing"] == 0
    assert rec["bucket_columns"] == BUCKETS and rec["n_buckets"] == 2
    assert math.isclose(rec["mean_pred"], 0.52) and math.isclose(rec["mean_realized"], 0.52)
    assert math.isclose(rec["signed_bias"], 0.0, abs_tol=1e-12)
    b10, b20 = rec["buckets"]  # sorted by the bucket columns
    assert (b10["p_bp"], b10["q_usd"], b10["h_ms"], b10["n"]) == (10, 1000, 60_000, 2)
    assert math.isclose(b10["mean_pred"], 0.7) and math.isclose(b10["mean_realized"], 1.0)
    assert math.isclose(b10["bias"], -0.3)
    assert (b20["p_bp"], b20["n"]) == (20, 3)
    assert math.isclose(b20["mean_pred"], 0.4) and math.isclose(b20["mean_realized"], 0.2)
    assert math.isclose(b20["bias"], 0.2)
    assert rec["max_abs_bucket_bias"] == b10  # the largest |bias| is the negative one
    # one batch or two: the same record
    one = CalibrationBias(BUCKETS)
    one.add(
        _keys([10, 20, 10, 20, 20]), np.array([0.8, 0.5, 0.6, 0.4, 0.3], dtype=np.float32),
        np.array([1.0, 0.5, 1.0, 0.0, 0.1]),
    )  # fmt: skip
    got = one.record()
    assert [(b["p_bp"], b["n"]) for b in got["buckets"]] == [(10, 2), (20, 3)]
    # float32 predictions are read as float64 of the same values
    f32 = np.array([0.8, 0.6], dtype=np.float32).astype(np.float64).mean()
    assert math.isclose(got["buckets"][0]["mean_pred"], float(f32), rel_tol=1e-15)


def test_rows_without_a_realized_value_are_dropped_and_an_all_missing_bucket_is_absent() -> None:
    nan = float("nan")
    acc = CalibrationBias(BUCKETS)
    acc.add(_keys([10, 10, 30, 30]), np.array([0.2, 0.4, 0.9, 0.1]), np.array([0.0, nan, nan, nan]))
    rec = acc.record()
    assert rec["n"] == 1 and rec["n_realized_missing"] == 3
    # p 30 has no realized row: no bucket -- never a bias of nothing
    assert [b["p_bp"] for b in rec["buckets"]] == [10] and rec["n_buckets"] == 1
    assert math.isclose(rec["signed_bias"], 0.2)
    assert rec["max_abs_bucket_bias"] == rec["buckets"][0]


def test_no_bucket_columns_report_the_overall_bias_alone() -> None:
    acc = CalibrationBias([])
    acc.add(None, np.array([0.5, 0.7]), np.array([1.0, 0.0]))
    rec = acc.record()
    assert math.isclose(rec["signed_bias"], 0.1)  # 0.6 - 0.5
    assert rec["buckets"] == [] and rec["n_buckets"] == 0 and rec["max_abs_bucket_bias"] is None


def test_no_row_reports_no_bias_rather_than_a_number() -> None:
    acc = CalibrationBias(BUCKETS)
    rec = acc.record()  # nothing added: an empty partition
    assert rec["n"] == 0 and rec["mean_pred"] is None and rec["mean_realized"] is None
    assert (
        rec["signed_bias"] is None and rec["buckets"] == [] and rec["max_abs_bucket_bias"] is None
    )
    acc.add(_keys([10]), np.array([0.5]), np.array([float("nan")]))  # every realized missing
    rec = acc.record()
    assert rec["n"] == 0 and rec["n_realized_missing"] == 1 and rec["signed_bias"] is None
    assert rec["buckets"] == [] and rec["n_buckets"] == 0


def test_calibration_refuses_what_it_cannot_score() -> None:
    acc = CalibrationBias(BUCKETS)
    with pytest.raises(ValueError, match="prediction"):
        acc.add(_keys([10]), np.array([float("nan")]), np.array([1.0]))
    with pytest.raises(ValueError, match="shape"):
        acc.add(_keys([10, 10]), np.array([0.5]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="h_ms"):
        acc.add(pl.DataFrame({"p_bp": [10], "q_usd": [1000]}), np.array([0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="rows"):
        acc.add(_keys([10, 20]), np.array([0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="bucket frame"):
        acc.add(None, np.array([0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="collide"):
        CalibrationBias(["p_bp", "n"])  # a bucket column named like the record's own keys
    with pytest.raises(ValueError, match="distinct"):
        CalibrationBias(["p_bp", "p_bp"])
    assert acc.record()["n"] == 0  # a refused batch added nothing
