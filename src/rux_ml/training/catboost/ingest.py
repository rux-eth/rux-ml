"""CatBoost ingest placeholder (per PR-019 Q-Wrap §9).

Unlike XGBoost (``QuantileDMatrix`` vs ``ExtMemQuantileDMatrix`` per D3) or
LightGBM (``Dataset`` with histogram binning per PR-018 Q-Ingest), CatBoost's
``CatBoostClassifier.fit(X, y)`` accepts a ``pandas.DataFrame`` (or
``polars.DataFrame`` per v1.2+, or numpy array) directly with no explicit
ingest-object construction required. ``catboost.Pool`` is the optional
analog of ``DMatrix`` / ``Dataset`` but is NOT needed for the workbench's
flow.

This module exists for symmetry with ``training/xgboost/ingest.py`` and
``training/lightgbm/ingest.py`` and as the home for any future CatBoost-side
ingest decision rule (e.g., explicit ``Pool`` construction for streaming
training, or memory-tight tier selection). For v0.1 it's a pass-through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from rux_ml.config import DataConfig


def build_pool(
    x: Any,
    y: ArrayLike,
    data_cfg: DataConfig,
    *,
    cat_features: Any = None,
) -> Any:
    """Build a CatBoost ``Pool`` from ``(x, y)``.

    Not called by ``make_catboost_trainer`` — the factory passes the raw
    DataFrame directly to ``CatBoostClassifier.fit``, which CatBoost accepts
    natively (Q-Wrap §9 PROVEN). Provided as a utility for advanced users
    who want explicit ``Pool`` construction (e.g., for ``baseline=``,
    ``weights=``, ``timestamp=``, or other Pool-only knobs).

    Args:
        x: Feature frame (pandas DataFrame, polars DataFrame, or numpy array).
        y: Target column.
        data_cfg: Data layer config. Currently unused — kept for signature
            symmetry with the xgboost/lightgbm ingest helpers.
        cat_features: List of categorical column names or indices. If None,
            CatBoost's auto-detect rejects pandas Categorical-dtype columns
            (Q-Cat finding — use ``_CatBoostTrainerShim`` which extracts at
            fit time, or pass explicitly here).

    Returns:
        A ``catboost.Pool`` instance.
    """
    _ = data_cfg  # reserved for future CatBoost-side ingest rules

    from catboost import Pool  # noqa: PLC0415 — lazy import (extra-gated)

    return Pool(data=x, label=y, cat_features=cat_features)
