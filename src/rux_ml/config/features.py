"""Feature-engineering layer config (per D4)."""

from typing import Any

from pydantic import Field

from rux_ml.config._strict_model import StrictModel


class FeaturesConfig(StrictModel):
    # BEST-GUESS per D4: no default value. Setting this on first use is a
    # deliberate decision — categorical handling differs above/below the
    # threshold (XGBoost `enable_categorical=True` vs `category_encoders`
    # NestedCVWrapper).
    categorical_low_card_threshold: int | None = None

    # Placeholder spec dict; PR-005 will narrow this into a typed schema.
    spec: dict[str, Any] = Field(default_factory=dict)
