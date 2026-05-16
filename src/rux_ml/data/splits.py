"""Deterministic train/val/test splits.

Per D9, a full ``SeedSequence`` + per-component spawn lands in PR-013;
this PR uses a single integer seed routed through Polars' native shuffle
so PR-013 can plug a derived seed in unchanged.

**One-shot vs repeated CV** (PR-015): ``train_val_test_split`` is a one-shot
convenience that returns Polars DataFrames directly. For repeated K-fold-style
cross-validation (used by PR-007's Optuna objective and any HPO loop), use the
``Splitter`` Protocol in :mod:`rux_ml.data.cv` instead — it yields ``(train_idx,
test_idx)`` row-index pairs (sklearn convention) so the caller controls
materialisation policy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

_TOL = 1e-6


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    if set(ratios) != {"train", "val", "test"}:
        msg = f"split ratios keys must be exactly train/val/test, got {sorted(ratios)}"
        raise ValueError(msg)
    if any(v < 0 for v in ratios.values()):
        msg = f"split ratios must be non-negative, got {dict(ratios)}"
        raise ValueError(msg)
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, abs_tol=_TOL):
        msg = f"split ratios must sum to 1.0, got {total:.6f} from {dict(ratios)}"
        raise ValueError(msg)


def train_val_test_split(
    df: pl.DataFrame,
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Shuffle rows deterministically by ``seed`` and slice into train/val/test.

    ``ratios`` must contain exactly the keys ``train``, ``val``, ``test`` and
    sum to 1.0 (within 1e-6 tolerance).
    """
    _validate_ratios(ratios)
    n = df.height
    if n == 0:
        return {k: df.clone() for k in ("train", "val", "test")}

    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    train_n = int(n * ratios["train"])
    val_n = int(n * ratios["val"])
    # Test gets whatever is left to ensure all rows are placed.
    test_n = n - train_n - val_n
    return {
        "train": shuffled.slice(0, train_n),
        "val": shuffled.slice(train_n, val_n),
        "test": shuffled.slice(train_n + val_n, test_n),
    }
