"""Categorical encoder factory implementing the D4 decision rule.

Decision rule (per `docs/ARCHITECTURE.md`):

    if cardinality(col) ≤ FeaturesConfig.categorical_low_card_threshold:
        pass through as Polars Categorical → native categorical handling
        (XGBoost enable_categorical=True; LightGBM categorical_feature="auto"
        auto-detection of pandas Categorical dtype — same downstream
        encoding-of-nothing sentinel for both families, per PR-018 Q-Cat)
    else:
        category_encoders TargetEncoder via NestedCVWrapper
        (or hash encoding if cardinality is truly massive — configurable)

The threshold has no default per D4 (BEST-GUESS, tune on first dataset).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from category_encoders import TargetEncoder
from category_encoders.wrapper import NestedCVWrapper
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from rux_ml.config import FeaturesConfig

# Sentinel — `make_categorical_encoder` returns this constant when the column
# is below the low-card threshold and should be passed through unchanged to
# whichever family-native categorical handling is downstream (XGBoost's
# enable_categorical=True; LightGBM's categorical_feature="auto" pandas-
# Categorical auto-detect). The ColumnTransformer wiring treats this as a
# literal "passthrough" instruction. Renamed from PASSTHROUGH_TO_XGB_CATEGORICAL
# in PR-018 because the same sentinel serves both families per Q-Cat research.
PASSTHROUGH_NATIVE_CATEGORICAL: str = "passthrough"


def make_categorical_encoder(
    column: str,
    cardinality: int,
    cfg: FeaturesConfig,
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> NestedCVWrapper | str:
    """Return the encoder for a categorical column per the D4 decision rule.

    Args:
        column: Name of the categorical column.
        cardinality: Distinct-value count for the column.
        cfg: Features layer config (provides ``categorical_low_card_threshold``).
        n_splits: K-fold split count for ``NestedCVWrapper`` (used only for high-card columns).
        random_state: Seed for ``NestedCVWrapper`` shuffling.

    Returns:
        ``PASSTHROUGH_NATIVE_CATEGORICAL`` (the string ``"passthrough"``) for low-cardinality
        columns, signaling the caller to skip encoding (Polars Categorical reaches XGBoost
        directly). Otherwise a ``NestedCVWrapper(TargetEncoder(...))`` instance.

    Raises:
        ValueError: if ``cfg.categorical_low_card_threshold`` is ``None`` (D4 BEST-GUESS:
            tuned on first dataset; setting it must be deliberate).
    """
    threshold = cfg.categorical_low_card_threshold
    if threshold is None:
        msg = (
            "features.categorical_low_card_threshold has no default (BEST-GUESS per D4); "
            f"set it before encoding categorical column {column!r}"
        )
        raise ValueError(msg)
    if cardinality <= threshold:
        return PASSTHROUGH_NATIVE_CATEGORICAL
    return NestedCVWrapper(
        feature_encoder=TargetEncoder(cols=[column], handle_unknown="value"),
        cv=n_splits,
        shuffle=True,
        random_state=random_state,
    )


class _PolarsToFrame(BaseEstimator, TransformerMixin):
    """Lossless Polars ↔ pandas shim for legacy sklearn-API transformers.

    ``category_encoders`` predates sklearn 1.4's native Polars support, so we
    convert at the encoder boundary and convert back. Used only inside the
    encoder branch of the ColumnTransformer; pure Polars steps don't touch it.
    """

    def __init__(self) -> None:
        pass

    def fit(self, x: Any, y: Any = None) -> _PolarsToFrame:
        _ = (x, y)
        return self

    def transform(self, x: Any) -> Any:
        import pandas as pd  # noqa: PLC0415
        import polars as pl  # noqa: PLC0415

        if isinstance(x, pl.DataFrame):
            return x.to_pandas()
        if isinstance(x, pd.DataFrame):
            return x
        msg = f"_PolarsToFrame expected pl.DataFrame or pd.DataFrame, got {type(x).__name__}"
        raise TypeError(msg)
