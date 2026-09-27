"""The signed-error honesty test for a predicted cost (PR-044; program v0.3 D41, D45 #3).

Program D41: "(i) no under-deduction: predicted cost >= realized on at least a
stated fraction of rows, (ii) a bounded over-deduction, reported, (iii) both on
every regime; rux-ml's default metric reported beside it". The fraction and the
bound are operator-signed keys (:mod:`rux_ml.config.m9_gates`), passed in.

Both inputs are **costs in bp** (higher = worse); the sign convention of the
label is the program's. The over-deduction statistic is the net relative bias
``(mean(pred) - mean(realized)) / |mean(realized)|`` — the program docs do not
define the aggregation (gen-2's survivor "held +22-25 %"), so this definition is
this PR's, flagged for the program lead. A zero mean realized cost leaves it
undefined: reported ``None`` and never a pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def signed_error_honesty(
    pred: ArrayLike,
    realized: ArrayLike,
    *,
    no_underdeduct_frac_min: float,
    overdeduct_max_rel: float,
) -> dict[str, Any]:
    """Score predicted costs against realized costs; rows with no realized cost are
    dropped and counted. A missing prediction, or no row left, is refused."""
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
    frac = float(np.mean(p >= r))
    mean_real = float(np.mean(r))
    over = None if mean_real == 0.0 else (float(np.mean(p)) - mean_real) / abs(mean_real)
    passes_under = frac >= no_underdeduct_frac_min
    passes_over = over is not None and over <= overdeduct_max_rel
    return {
        "n": n,
        "n_realized_missing": int((~keep).sum()),
        "no_underdeduct_frac": frac,
        "no_underdeduct_frac_min": no_underdeduct_frac_min,
        "overdeduct_rel": over,
        "overdeduct_max_rel": overdeduct_max_rel,
        "mae_bp": float(np.mean(np.abs(err))),
        "mean_signed_error_bp": float(np.mean(err)),
        "passes_no_underdeduct": bool(passes_under),
        "passes_overdeduct": bool(passes_over),
        "honest": bool(passes_under and passes_over),
    }
