"""The M9 honesty records (PR-044, PR-058; program v0.3 D41, D45 #3, PR-027 A7).

**The signed-error honesty test** (PR-044). Program D41: "(i) no under-deduction:
predicted cost >= realized on at least a stated fraction of rows, (ii) a bounded
over-deduction, reported, (iii) both on every regime; rux-ml's default metric reported
beside it". The fraction and the bound are operator-signed keys
(:mod:`rux_ml.config.m9_gates`), passed in.

Both inputs are **costs in bp** (higher = worse); the sign convention of the
label is the program's. The over-deduction statistic is the net relative bias
``(mean(pred) - mean(realized)) / |mean(realized)|`` — the program docs did not
define the aggregation (gen-2's survivor "held +22-25 %"), so the definition was
PR-044's, flagged for the program lead; program PR-027 A7 kept it ("Q5c: keep the
signed definition, no re-signing"). A zero mean realized cost leaves it undefined:
reported ``None`` and never a pass.

**The two forms** (PR-058; program PR-027 A7, operator-approved 2026-09-28: "Markout
is tested as the EV deducts it, max(0, m) < r"). The test runs on two forms of the
prediction, both always reported under ``forms``: ``signed`` — the prediction as
fitted (PR-044's) — and ``clipped_at_zero`` — ``max(0, pred)``, the cost the
program's EV deducts. ``verdict_form`` names the form whose values the verdict keys
(``no_underdeduct_frac``, ``overdeduct_rel``, ``passes_*``, ``honest``) carry; the
problem config chooses it (``[m9] honesty_verdict_form``, default ``signed``).

**The signed calibration bias** (PR-058; A7: "fill is gated on Brier, with a signed
calibration bias per (p, Q, h) bucket reported"): ``mean(pred) - mean(realized)``
overall and per bucket — each distinct value tuple of the configured bucket columns —
with the bucket of the largest ``|bias|``; accumulated batch by batch
(:class:`CalibrationBias`). Reported, never gated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import polars as pl

from rux_ml.config.m9 import HonestyForm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

HONESTY_FORMS: tuple[HonestyForm, ...] = get_args(HonestyForm)


def _form(
    p: NDArray[np.float64],
    r: NDArray[np.float64],
    mean_real: float,
    no_underdeduct_frac_min: float,
    overdeduct_max_rel: float,
) -> dict[str, Any]:
    """PR-044's two criteria for one form of the prediction."""
    frac = float(np.mean(p >= r))
    mean_pred = float(np.mean(p))
    over = None if mean_real == 0.0 else (mean_pred - mean_real) / abs(mean_real)
    passes_under = frac >= no_underdeduct_frac_min
    passes_over = over is not None and over <= overdeduct_max_rel
    return {
        "no_underdeduct_frac": frac,
        "overdeduct_rel": over,
        "mean_pred_bp": mean_pred,
        "passes_no_underdeduct": bool(passes_under),
        "passes_overdeduct": bool(passes_over),
        "honest": bool(passes_under and passes_over),
    }


def signed_error_honesty(
    pred: ArrayLike,
    realized: ArrayLike,
    *,
    no_underdeduct_frac_min: float,
    overdeduct_max_rel: float,
    verdict_form: HonestyForm = "signed",
) -> dict[str, Any]:
    """Score predicted costs against realized costs in both forms; rows with no realized
    cost are dropped and counted. A missing prediction, no row left, or an unknown
    ``verdict_form`` is refused."""
    if verdict_form not in HONESTY_FORMS:
        msg = f"honesty: verdict_form {verdict_form!r} is not one of {HONESTY_FORMS}"
        raise ValueError(msg)
    p = np.asarray(pred, dtype=float)
    r = np.asarray(realized, dtype=float)
    if p.shape != r.shape:
        msg = f"honesty: prediction shape {p.shape} != realized shape {r.shape}"
        raise ValueError(msg)
    if np.isnan(p).any():
        msg = "honesty: a prediction is missing (NaN)"
        raise ValueError(msg)
    keep = ~np.isnan(r)
    p, r = p[keep], r[keep]
    n = int(p.size)
    if n == 0:
        msg = "honesty: no row with a realized cost"
        raise ValueError(msg)
    err = p - r
    mean_real = float(np.mean(r))
    bounds = (no_underdeduct_frac_min, overdeduct_max_rel)
    forms = {
        "signed": _form(p, r, mean_real, *bounds),
        # A7: the cost the EV deducts is max(0, m) — the clip is the EV's definition.
        "clipped_at_zero": _form(np.maximum(p, 0.0), r, mean_real, *bounds),
    }
    verdict = forms[verdict_form]
    return {
        "n": n,
        "n_realized_missing": int((~keep).sum()),
        "no_underdeduct_frac_min": no_underdeduct_frac_min,
        "overdeduct_max_rel": overdeduct_max_rel,
        "mean_realized_bp": mean_real,
        # the fitted (signed) prediction's error, whichever form decides
        "mae_bp": float(np.mean(np.abs(err))),
        "mean_signed_error_bp": float(np.mean(err)),
        "verdict_form": verdict_form,
        "no_underdeduct_frac": verdict["no_underdeduct_frac"],
        "overdeduct_rel": verdict["overdeduct_rel"],
        "passes_no_underdeduct": verdict["passes_no_underdeduct"],
        "passes_overdeduct": verdict["passes_overdeduct"],
        "honest": verdict["honest"],
        "forms": forms,
    }


# The per-bucket record's own keys: a bucket column may not share a name with them.
_BUCKET_STATS = ("n", "mean_pred", "mean_realized", "bias")
_SUMS = ("n", "sum_pred", "sum_real")


class CalibrationBias:
    """The signed calibration bias ``mean(pred) - mean(realized)``, overall and per bucket
    of ``bucket_columns``, accumulated batch by batch: each :meth:`add` keeps per-bucket
    sums only, so no row is held. Rows without a realized value are dropped and counted,
    as the honesty test drops them; a bucket with no realized row is absent, never a bias
    of nothing. No row at all reports ``None``, never a number."""

    def __init__(self, bucket_columns: Sequence[str]) -> None:
        cols = list(bucket_columns)
        if len(set(cols)) != len(cols):
            msg = f"calibration: bucket columns must be distinct, got {cols}"
            raise ValueError(msg)
        clash = sorted(set(cols) & {*_BUCKET_STATS, *_SUMS})
        if clash:
            msg = f"calibration: bucket column(s) {clash} collide with the record's keys"
            raise ValueError(msg)
        self.bucket_columns = cols
        self._n = 0
        self._n_missing = 0
        self._sum_pred = 0.0
        self._sum_real = 0.0
        self._parts: list[pl.DataFrame] = []

    def add(self, buckets: pl.DataFrame | None, pred: ArrayLike, realized: ArrayLike) -> None:
        """One batch: its bucket columns (``None`` without bucket columns), predictions
        and realized values, row-aligned. A refused batch adds nothing."""
        p = np.asarray(pred, dtype=np.float64)
        r = np.asarray(realized, dtype=np.float64)
        if p.shape != r.shape or p.ndim != 1:
            msg = f"calibration: prediction shape {p.shape} != realized shape {r.shape}"
            raise ValueError(msg)
        if np.isnan(p).any():
            msg = "calibration: a prediction is missing (NaN)"
            raise ValueError(msg)
        keep = ~np.isnan(r)
        part = None
        if self.bucket_columns:
            if buckets is None:
                msg = f"calibration: bucket columns {self.bucket_columns} but no bucket frame"
                raise ValueError(msg)
            missing = [c for c in self.bucket_columns if c not in buckets.columns]
            if missing:
                msg = f"calibration: bucket column(s) {missing} not in the batch"
                raise ValueError(msg)
            if buckets.height != p.size:
                msg = (
                    f"calibration: the bucket frame has {buckets.height} rows, predictions {p.size}"
                )
                raise ValueError(msg)
            part = (
                buckets.select(self.bucket_columns)
                .with_columns(pl.Series("sum_pred", p), pl.Series("sum_real", r))
                .filter(pl.Series(keep))
                .group_by(self.bucket_columns)
                .agg(
                    pl.len().cast(pl.Int64).alias("n"),
                    pl.col("sum_pred").sum(),
                    pl.col("sum_real").sum(),
                )
            )
        self._n_missing += int((~keep).sum())
        self._n += int(keep.sum())
        self._sum_pred += float(p[keep].sum())
        self._sum_real += float(r[keep].sum())
        if part is not None:
            self._parts.append(part)

    def record(self) -> dict[str, Any]:
        """The overall and per-bucket bias (buckets sorted by the bucket columns) and the
        bucket of the largest ``|bias|`` (the first in that order on a tie)."""
        n = self._n
        mean_pred = self._sum_pred / n if n else None
        mean_real = self._sum_real / n if n else None
        buckets: list[dict[str, Any]] = []
        if self._parts:
            b = (
                pl.concat(self._parts)
                .group_by(self.bucket_columns)
                .agg(pl.col(c).sum() for c in _SUMS)
                .filter(pl.col("n") > 0)
                .with_columns(
                    (pl.col("sum_pred") / pl.col("n")).alias("mean_pred"),
                    (pl.col("sum_real") / pl.col("n")).alias("mean_realized"),
                )
                .with_columns((pl.col("mean_pred") - pl.col("mean_realized")).alias("bias"))
                .sort(self.bucket_columns)
                .select(*self.bucket_columns, *_BUCKET_STATS)
            )
            buckets = b.to_dicts()
        worst = max(buckets, key=lambda d: abs(d["bias"])) if buckets else None
        return {
            "n": n,
            "n_realized_missing": self._n_missing,
            "mean_pred": mean_pred,
            "mean_realized": mean_real,
            "signed_bias": None if n == 0 else mean_pred - mean_real,  # type: ignore[operator]
            "bucket_columns": list(self.bucket_columns),
            "n_buckets": len(buckets),
            "buckets": buckets,
            "max_abs_bucket_bias": worst,
        }
