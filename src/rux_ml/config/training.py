"""Training layer config — discriminated union over per-family variants (per PR-017).

Re-exports the family-agnostic :class:`TrainingBase` (from
``rux_ml.training.base``) and the XGBoost variant :class:`XGBoostTraining`
(from ``rux_ml.training.xgboost.config``) so existing callers' imports
(``from rux_ml.config import TrainingConfig``) continue to resolve. The
discriminated-union :data:`TrainingConfig` is assembled here at the
config-layer level.

The narrow layer inversion (``rux_ml.config`` depending on
``rux_ml.training.xgboost.config``) is intentional: Pydantic v2 discriminated
unions need the assembly point to know all variants by type. Per-family
variants live co-located with their factories per Q5 of
``docs/0.1/DESIGN-log.md`` — they're discovered by the registry at runtime
and assembled into the union here at type level.

PR-018/PR-019 widen the union by adding ``LightGBMTraining`` / ``CatBoostTraining``
imports + the corresponding variants in the ``Annotated[...]`` below.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from rux_ml.training.base import TrainingBase
from rux_ml.training.lightgbm.config import LightGBMTraining
from rux_ml.training.xgboost.config import XGBoostTraining

# Discriminated-union TrainingConfig. PR-017 introduced the pattern with
# XGBoostTraining; PR-018 widens to include LightGBMTraining. Pydantic v2
# dispatches on the ``kind`` discriminator field.
TrainingConfig = Annotated[
    XGBoostTraining | LightGBMTraining,
    Field(discriminator="kind"),
]

__all__ = [
    "LightGBMTraining",
    "TrainingBase",
    "TrainingConfig",
    "XGBoostTraining",
]
