"""Polars-expression helpers for the stateless side of the feature pipeline.

Wrapped via ``sklearn.preprocessing.FunctionTransformer`` so the whole pipeline
stays a single ``sklearn.Pipeline``.

For v0 the only stateless step is column selection + Polars Categorical casting
for low-cardinality categoricals. Derived columns (ratios, datetime parts,
aggregates) per D4 are a deferred extension to ``FeaturesSpec``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from rux_ml.config import FeaturesSpec


def select_columns(df: pl.DataFrame, spec: FeaturesSpec) -> pl.DataFrame:
    """Return only the spec's numeric + categorical columns, in that order.

    Categorical columns are cast to ``pl.Categorical`` so downstream XGBoost
    ``enable_categorical=True`` can consume them directly (per D4 decision
    rule for low-cardinality columns).
    """
    keep = [*spec.numeric_columns, *spec.categorical_columns]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        msg = f"FeaturesSpec references missing columns: {missing}"
        raise KeyError(msg)
    casts = [
        pl.col(c).cast(pl.Categorical)
        for c in spec.categorical_columns
        if df.schema[c] != pl.Categorical
    ]
    if casts:
        df = df.with_columns(casts)
    return df.select(keep)
