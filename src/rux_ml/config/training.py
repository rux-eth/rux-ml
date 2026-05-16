"""Training layer config (per D5, D3)."""

from typing import Any, Literal

from pydantic import Field

from rux_ml.config._strict_model import StrictModel


class TrainingConfig(StrictModel):
    # Trainer family (per D5 — sklearn estimator API as the contract).
    kind: Literal["xgboost", "lightgbm", "catboost", "sklearn"] = "xgboost"

    # GPU-first per D3 (user override of CPU-first lean).
    device: Literal["cuda", "cpu"] = "cuda"

    # Eval metric (concrete metric registry lands in PR-006).
    metric: str = "auc"

    # XGBoost-specific defaults that match D3 / D4 decisions.
    enable_categorical: bool = True
    tree_method: Literal["hist", "approx", "exact"] = "hist"

    # Native xgb.train() escape hatch per D5 (~5% of cases).
    use_native: bool = False

    # Common tunable hyperparameters surfaced at the top level so dot-path
    # CLI/env overrides stay clean (e.g. RUXML_TRAINING__LEARNING_RATE=0.01).
    learning_rate: float = 0.1
    max_depth: int = 6
    n_estimators: int = 100
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    early_stopping_rounds: int | None = 50

    # Anything else (XGBoost-specific or family-specific kwargs).
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
