"""Training layer — sklearn-API ``Trainer`` Protocol + XGBoost factory + metric registry.

Per D5: the contract across model families is the sklearn estimator API
expressed as a ``typing.Protocol``. Per D3: the ingest-path selector chooses
``QuantileDMatrix`` vs ``ExtMemQuantileDMatrix`` based on estimated X size.
"""

from rux_ml.training.factory import make_trainer
from rux_ml.training.ingest import (
    DEFAULT_BYTES_PER_GB,
    estimate_x_bytes,
    select_ingest,
)
from rux_ml.training.metrics import (
    METRIC_REGISTRY,
    compute_score,
    optuna_direction,
    task_for_metric,
)
from rux_ml.training.protocol import Trainer

__all__ = [
    "DEFAULT_BYTES_PER_GB",
    "METRIC_REGISTRY",
    "Trainer",
    "compute_score",
    "estimate_x_bytes",
    "make_trainer",
    "optuna_direction",
    "select_ingest",
    "task_for_metric",
]
