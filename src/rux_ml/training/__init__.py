"""Training layer — multi-family registry + sklearn-API ``Trainer`` Protocol.

Per PR-017 / docs/0.1/DESIGN-log.md Q4: every Trainer family registers itself
in :data:`TRAINER_FAMILIES`, a plain ``dict[str, Callable]`` mapping family
name (the ``kind`` discriminator) to a factory function. The top-level
:func:`make_trainer` dispatches against this dict.

The B-explicit pattern (HF transformers ``MODEL_MAPPING_NAMES``) is used
rather than a decorator-driven registry because at the workbench's scale
(3-5 families expected) the explicit dict is leaner, statically analyzable,
and has no decoration-order concerns.

Stale-path tripwire: ``tests/training/test_registry_conformance.py``
parametrizes over ``TRAINER_FAMILIES.keys()`` so any registered family that
fails to satisfy the :class:`Trainer` Protocol or fails an end-to-end
fit-predict smoke test fails CI loudly. Family removal = delete the dict
entry + delete the subpackage; orphan references show up as failing tests.

Per D3: the ingest-path selector chooses ``QuantileDMatrix`` vs
``ExtMemQuantileDMatrix`` based on estimated X size. The selector is
XGBoost-specific and now lives in ``training.xgboost.ingest``; it's re-exported
here for back-compat (existing callers in ``cli/train.py`` and
``tuning/objective.py`` need not update their imports in PR-017).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rux_ml.training.base import TrainingBase
from rux_ml.training.factory import make_trainer
from rux_ml.training.metrics import (
    METRIC_REGISTRY,
    compute_score,
    optuna_direction,
    task_for_metric,
)
from rux_ml.training.protocol import Trainer
from rux_ml.training.xgboost import (
    DEFAULT_BYTES_PER_GB,
    XGBoostTraining,
    estimate_x_bytes,
    make_xgboost_trainer,
    select_ingest,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# B-explicit registry (per Q4 of docs/0.1/DESIGN-log.md). New families register
# themselves here in their landing PR. Order is preserved (Python 3.7+ dict
# insertion order); ``test_registry_conformance.py`` iterates this in sorted
# key order so test parametrization is stable across CI runs regardless of
# insertion order.
TRAINER_FAMILIES: dict[str, Callable[..., Trainer]] = {
    "xgboost": make_xgboost_trainer,
}

__all__ = [
    "DEFAULT_BYTES_PER_GB",
    "METRIC_REGISTRY",
    "TRAINER_FAMILIES",
    "Trainer",
    "TrainingBase",
    "XGBoostTraining",
    "compute_score",
    "estimate_x_bytes",
    "make_trainer",
    "make_xgboost_trainer",
    "optuna_direction",
    "select_ingest",
    "task_for_metric",
]
