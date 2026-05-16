"""Feature-engineering layer config (per D4 + PR-005 narrowing)."""

from pydantic import Field

from rux_ml.config._strict_model import StrictModel


class FeaturesSpec(StrictModel):
    """Typed feature spec consumed by ``rux_ml.features.make_features``.

    PR-005 narrows the previous ``dict[str, Any]`` placeholder into this typed
    schema. Stays minimal for v0 — derived columns and stateful sklearn steps
    can be added when a concrete use case demands them.
    """

    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)


class FeaturesConfig(StrictModel):
    # BEST-GUESS per D4: no default value. Setting this on first use is a
    # deliberate decision — categorical handling differs above/below the
    # threshold (XGBoost `enable_categorical=True` vs `category_encoders`
    # NestedCVWrapper).
    categorical_low_card_threshold: int | None = None

    spec: FeaturesSpec = Field(default_factory=FeaturesSpec)
