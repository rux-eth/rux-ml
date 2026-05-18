"""Features layer — Polars (stateless) + sklearn Pipeline/ColumnTransformer (stateful).

Per D4: stateless transforms live in Polars expressions; stateful transforms
(target encoding etc.) live in sklearn; the whole thing is one ``Pipeline``
object via ``set_output("polars")``.
"""

from rux_ml.features.encoders import (
    PASSTHROUGH_NATIVE_CATEGORICAL,
    make_categorical_encoder,
)
from rux_ml.features.pipeline import (
    build_column_transformer,
    cardinalities_from,
    make_features,
)
from rux_ml.features.polars_steps import select_columns

__all__ = [
    "PASSTHROUGH_NATIVE_CATEGORICAL",
    "build_column_transformer",
    "cardinalities_from",
    "make_categorical_encoder",
    "make_features",
    "select_columns",
]
