"""``make_features(cfg)`` factory — composes a single sklearn Pipeline.

Architecture per D4:
- **Stateless stage:** Polars-side column selection + Categorical casting, then
  conversion to pandas so legacy sklearn-API transformers (notably
  ``category_encoders.NestedCVWrapper``, which predates sklearn 1.4's
  ``set_output("polars")``) can work natively.
- **Stateful stage:** :class:`_ColumnRouter` routes each categorical column
  through the encoder returned by
  :func:`rux_ml.features.encoders.make_categorical_encoder`. Numeric columns
  pass through unchanged. Built as a small custom transformer (rather than
  ``sklearn.compose.ColumnTransformer``) because ColumnTransformer's
  ``set_output`` propagation breaks on ``NestedCVWrapper``, and because we
  need an explicit ``fit_transform`` path so ``NestedCVWrapper`` produces its
  un-leaked training-set encodings.
- **Polars-back stage:** convert pandas → Polars so downstream
  (training layer, registry) stays in Polars-land with column names preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from rux_ml.features.encoders import (
    PASSTHROUGH_TO_XGB_CATEGORICAL,
    make_categorical_encoder,
)
from rux_ml.features.polars_steps import select_columns

if TYPE_CHECKING:
    from rux_ml.config import FeaturesConfig, FeaturesSpec


def _polars_select_then_pandas(df: pl.DataFrame, spec: FeaturesSpec) -> pd.DataFrame:
    """Stateless Polars block, returning pandas so legacy sklearn transformers work.

    Takes ``spec`` (not the full ``FeaturesConfig``) as a kw-arg so the function
    stays module-level — required for ``skops.io`` serialization of the
    surrounding ``FunctionTransformer`` (per PR-010 bundle round-trip).
    """
    return select_columns(df, spec).to_pandas()


def _to_polars(x: Any) -> pl.DataFrame:
    """Final boundary: pandas DataFrame → Polars DataFrame, preserving column names."""
    if isinstance(x, pl.DataFrame):
        return x
    if isinstance(x, pd.DataFrame):
        return pl.from_pandas(x)
    msg = f"_to_polars expected pl.DataFrame or pd.DataFrame, got {type(x).__name__}"
    raise TypeError(msg)


class _ColumnRouter(BaseEstimator, TransformerMixin):
    """Per-column transformer routing that preserves column names without ColumnTransformer.

    For each categorical column either passes it through (low-card → XGBoost
    native handling) or applies a fitted encoder (high-card → NestedCV-wrapped
    target encoder). Numeric columns always pass through.

    Critically uses ``fit_transform`` so ``NestedCVWrapper`` returns its
    un-leaked training encodings (separate ``fit().transform()`` on the same
    data would yield leaky encodings).
    """

    def __init__(
        self,
        spec: FeaturesSpec,
        encoders: dict[str, Any],
    ) -> None:
        self.spec = spec
        self.encoders = encoders

    @staticmethod
    def _col_frame(x: pd.DataFrame, col: str) -> pd.DataFrame:
        """Single-column slice as a DataFrame. Cast satisfies pandas typing
        (``df[[col]]`` is statically inferred as ``Series | DataFrame``)."""
        return cast("pd.DataFrame", x[[col]])

    def fit(self, X: pd.DataFrame, y: Any = None) -> _ColumnRouter:
        for col, enc in self.encoders.items():
            if enc != PASSTHROUGH_TO_XGB_CATEGORICAL:
                enc.fit(self._col_frame(X, col), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for col in self.spec.numeric_columns:
            parts.append(self._col_frame(X, col))
        for col in self.spec.categorical_columns:
            enc = self.encoders[col]
            if enc == PASSTHROUGH_TO_XGB_CATEGORICAL:
                parts.append(self._col_frame(X, col))
            else:
                out = enc.transform(self._col_frame(X, col))
                parts.append(self._as_named_frame(out, col, X.index))
        return pd.concat(parts, axis=1)

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> pd.DataFrame:
        _ = fit_params
        parts: list[pd.DataFrame] = []
        for col in self.spec.numeric_columns:
            parts.append(self._col_frame(X, col))
        for col in self.spec.categorical_columns:
            enc = self.encoders[col]
            if enc == PASSTHROUGH_TO_XGB_CATEGORICAL:
                parts.append(self._col_frame(X, col))
            else:
                # Critical: fit_transform on the wrapper, not fit+transform —
                # NestedCVWrapper uses K-fold to avoid within-train leakage
                # only on the fit_transform path.
                out = enc.fit_transform(self._col_frame(X, col), y)
                parts.append(self._as_named_frame(out, col, X.index))
        return pd.concat(parts, axis=1)

    @staticmethod
    def _as_named_frame(out: Any, col: str, idx: pd.Index) -> pd.DataFrame:
        if isinstance(out, pd.DataFrame):
            return out
        # numpy ndarray — wrap with the original column name + train index
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(out)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr, columns=[col], index=idx)


def build_column_transformer(
    cardinalities: dict[str, int],
    cfg: FeaturesConfig,
) -> _ColumnRouter:
    """Construct the per-column router per the D4 decision rule.

    Args:
        cardinalities: Per-column distinct-value counts. Required for every
            column in ``cfg.spec.categorical_columns``; ignored for numeric.
        cfg: Features layer config.
    """
    spec = cfg.spec
    missing = [c for c in spec.categorical_columns if c not in cardinalities]
    if missing:
        msg = f"cardinalities missing entries for categorical columns: {missing}"
        raise ValueError(msg)

    encoders = {
        col: make_categorical_encoder(col, cardinalities[col], cfg)
        for col in spec.categorical_columns
    }
    return _ColumnRouter(spec=spec, encoders=encoders)


def make_features(
    cfg: FeaturesConfig,
    cardinalities: dict[str, int] | None = None,
) -> Pipeline:
    """Build the feature pipeline.

    Pipeline shape (Polars in → Polars out, with a pandas waist):

        polars_select  →  column_router  →  to_polars

    Args:
        cfg: Features layer config.
        cardinalities: Per-categorical-column distinct-value counts. Required
            when ``cfg.spec.categorical_columns`` is non-empty. Callers
            typically compute this from the training Polars DataFrame
            via :func:`cardinalities_from`.

    Returns:
        A ``Pipeline`` whose ``fit_transform`` / ``transform`` return a Polars
        DataFrame with original column names preserved.
    """
    if cfg.spec.categorical_columns and cardinalities is None:
        msg = (
            "cardinalities is required when FeaturesSpec.categorical_columns "
            "is non-empty (used by the D4 categorical decision rule)"
        )
        raise ValueError(msg)
    router = build_column_transformer(cardinalities or {}, cfg)

    return Pipeline(
        steps=[
            (
                "polars_select",
                # `func` is module-level + `kw_args` carries ``spec`` so skops can
                # round-trip the FunctionTransformer (closures aren't serializable).
                FunctionTransformer(
                    func=_polars_select_then_pandas,
                    kw_args={"spec": cfg.spec},
                    validate=False,
                ),
            ),
            ("column_router", router),
            ("to_polars", FunctionTransformer(func=_to_polars, validate=False)),
        ]
    )


def cardinalities_from(df: pl.DataFrame, columns: list[str]) -> dict[str, int]:
    """Convenience: compute per-column distinct counts for ``columns``."""
    return {c: int(df[c].n_unique()) for c in columns}
