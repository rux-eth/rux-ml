"""PR-058 (program PR-027 A7): an M9 fit's record carries the honesty forms A7 names.

A7 (program PR-027 :690-691, operator-approved 2026-09-28): "fill is gated on Brier, with a
signed calibration bias per (p, Q, h) bucket reported [...]. Markout is tested as the EV
deducts it, max(0, m) < r." The batch-path tests hold ``fold_meta.json`` ``oos`` to the same
statistics computed independently from the fit's own exported test rows (PR-055's
``oos_rows.parquet``: the row keys, the prediction scored and the realized target) -- the
computation program PR-027 P2 made outside rux-ml. The in-memory record (the path an
``[m9]`` classification fit keeps) is held to hand-computed values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from rux_ml.cli import app
from rux_ml.cli.train import _oos_record  # pyright: ignore[reportPrivateUsage]
from rux_ml.config import RuxMLConfig
from rux_ml.config.m9_gates import M9Gates
from rux_ml.training.honesty import signed_error_honesty
from tests.cli.conftest import REPO, m9_argv
from tests.cli.test_m9_oos_export import _export  # pyright: ignore[reportPrivateUsage]
from tests.conftest import M9_GATE_VALUES, m9_gates_overrides

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import CliRunner

BUCKETS = ["p_bp", "q_usd", "h_ms"]
VERDICT_KEYS = (
    "no_underdeduct_frac", "overdeduct_rel", "passes_no_underdeduct", "passes_overdeduct", "honest",
)  # fmt: skip


def _test_rows(
    runner: CliRunner, tmp: Path, c6: Path, gates: dict[str, str], problem: str
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Train ``problem``; its exported test rows with a realized value, and its ``oos`` record."""
    argv = m9_argv(tmp, c6, problem, *m9_gates_overrides(gates), "train")
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    export, meta, _, _ = _export(tmp)
    r = pl.col("realized")
    test = export.filter((pl.col("partition") == "test") & r.is_not_null() & r.is_not_nan())
    return test.with_columns(pl.col("prediction").cast(pl.Float64)), meta["oos"]


def test_the_fill_record_carries_the_signed_calibration_bias_overall_and_per_bucket(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    test, oos = _test_rows(runner, tmp_path, c6_set, signed_m9_gates, "m9_fill_frac")
    assert "honesty" not in oos  # A7: Brier gates the fill; the bias is reported beside it
    cal = oos["calibration"]
    p, r = test["prediction"].to_numpy(), test["realized"].to_numpy()
    assert cal["n"] == test.height == oos["n_scored"] > 0
    assert math.isclose(cal["mean_pred"], float(np.mean(p)), rel_tol=1e-12)
    assert math.isclose(cal["mean_realized"], float(np.mean(r)), rel_tol=1e-12)
    assert math.isclose(cal["signed_bias"], float(np.mean(p) - np.mean(r)), abs_tol=1e-12)
    want = (
        test.group_by(BUCKETS)
        .agg(
            pl.len().alias("n"),
            pl.col("prediction").mean().alias("mean_pred"),
            pl.col("realized").mean().alias("mean_realized"),
        )
        .with_columns((pl.col("mean_pred") - pl.col("mean_realized")).alias("bias"))
        .sort(BUCKETS)
        .to_dicts()
    )
    assert cal["bucket_columns"] == BUCKETS and cal["n_buckets"] == len(want) > 1
    for got, exp in zip(cal["buckets"], want, strict=True):
        assert {k: got[k] for k in [*BUCKETS, "n"]} == {k: exp[k] for k in [*BUCKETS, "n"]}
        for k in ("mean_pred", "mean_realized", "bias"):
            assert math.isclose(got[k], exp[k], rel_tol=1e-9, abs_tol=1e-12)
    assert cal["max_abs_bucket_bias"] == max(cal["buckets"], key=lambda b: abs(b["bias"]))


def test_the_markout_record_carries_both_forms_and_its_verdict_is_the_clipped(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    test, oos = _test_rows(runner, tmp_path, c6_set, signed_m9_gates, "m9_markout_bp")
    h = oos["honesty"]
    p, r = test["prediction"].to_numpy(), test["realized"].to_numpy()
    assert (p < 0).any() and (p > 0).any()  # the clip bites on this fit
    pc = np.maximum(p, 0.0)
    signed, clipped = h["forms"]["signed"], h["forms"]["clipped_at_zero"]
    assert h["n"] == p.size == oos["n_scored"]
    assert signed["no_underdeduct_frac"] == float(np.mean(p >= r))
    assert clipped["no_underdeduct_frac"] == float(np.mean(pc >= r))
    mr = float(np.mean(r))
    assert math.isclose(signed["overdeduct_rel"], (float(np.mean(p)) - mr) / abs(mr), rel_tol=1e-9)
    assert math.isclose(
        clipped["overdeduct_rel"], (float(np.mean(pc)) - mr) / abs(mr), rel_tol=1e-9
    )
    # A7: the verdict is the form the EV deducts; the clip can only raise the fraction
    assert h["verdict_form"] == "clipped_at_zero"
    assert {k: h[k] for k in VERDICT_KEYS} == {k: clipped[k] for k in VERDICT_KEYS}
    assert clipped["no_underdeduct_frac"] >= signed["no_underdeduct_frac"]
    assert "calibration" not in oos


class _Identity:
    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        return x


class _ColumnModel:
    """Predicts the frame's ``stub`` column: a fit whose predictions are known."""

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(x["stub"], dtype=np.float32)


def _record(problem: str, target: str, stub: list[float], realized: list[float]) -> dict[str, Any]:
    cfg = RuxMLConfig.from_layers(
        REPO / "configs" / "base.toml", problem=problem, problems_dir=REPO / "configs" / "problems"
    )
    n = len(stub)
    test = pl.DataFrame(
        {"p_bp": [10, 10, 20, 20, 20][:n], "q_usd": [1000] * n, "h_ms": [60_000] * n,
         "stub": stub, target: realized}
    )  # fmt: skip
    gates = M9Gates(values=dict(M9_GATE_VALUES), sha256="0" * 64, path="gates.yaml")
    return _oos_record(cfg, _ColumnModel(), _Identity(), test, target, gates)


def test_the_in_memory_record_carries_the_calibration_bias() -> None:
    # p 10: 0.7 - 1.0 = -0.3; p 20: 0.4 - 0.2 = +0.2; overall 0.52 - 0.52 = 0
    stub, real = [0.8, 0.6, 0.5, 0.4, 0.3], [1.0, 1.0, 0.5, 0.0, 0.1]
    oos = _record("m9_fill_frac", "y__fill_frac", stub, real)
    cal = oos["calibration"]
    assert "honesty" not in oos and cal["n"] == 5 and cal["n_buckets"] == 2
    assert math.isclose(cal["signed_bias"], 0.0, abs_tol=1e-7)  # float32 predictions
    b10, b20 = cal["buckets"]
    assert math.isclose(b10["bias"], -0.3, rel_tol=1e-6) and math.isclose(
        b20["bias"], 0.2, rel_tol=1e-6
    )
    assert cal["max_abs_bucket_bias"] == b10


def test_the_in_memory_record_carries_both_forms_with_the_clipped_verdict() -> None:
    stub, real = [-2.0, 1.0, 3.0, -1.0, 0.5], [-1.0, 1.0, 2.0, 0.5, 1.0]
    h = _record("m9_markout_bp", "y__markout_bp", stub, real)["honesty"]
    want = signed_error_honesty(
        np.array(stub), np.array(real), verdict_form="clipped_at_zero",
        no_underdeduct_frac_min=M9_GATE_VALUES["m9_honesty_no_underdeduct_frac_min"],
        overdeduct_max_rel=M9_GATE_VALUES["m9_honesty_overdeduct_max_rel"],
    )  # fmt: skip
    assert h == want
    assert h["verdict_form"] == "clipped_at_zero" and h["no_underdeduct_frac"] == 0.6
    assert h["forms"]["signed"]["no_underdeduct_frac"] == 0.4
