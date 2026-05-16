"""Unit tests for ``make_trainer`` — concrete class + kwarg propagation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from xgboost import XGBClassifier, XGBModel, XGBRegressor

from rux_ml.config import TrainingConfig
from rux_ml.training import Trainer, make_trainer


def _accept(_trainer: Trainer) -> None:
    """Static-typing acceptance helper: basedpyright fails if ``XGBClassifier``
    does not conform to :class:`Trainer`. At runtime this is a no-op."""


def test_make_trainer_returns_xgbclassifier_for_classification_metric(
    training_cfg_classification: TrainingConfig,
) -> None:
    trainer = make_trainer(training_cfg_classification)
    assert isinstance(trainer, XGBClassifier)


def test_make_trainer_returns_xgbregressor_for_regression_metric(
    training_cfg_regression: TrainingConfig,
) -> None:
    trainer = make_trainer(training_cfg_regression)
    assert isinstance(trainer, XGBRegressor)


def _params(trainer: Trainer) -> dict[str, object]:
    """Narrow the universal ``Trainer`` Protocol to the sklearn-shape ``get_params``."""
    return cast("XGBModel", trainer).get_params()


def test_make_trainer_propagates_top_level_kwargs(
    training_cfg_classification: TrainingConfig,
) -> None:
    trainer = make_trainer(training_cfg_classification)
    params = _params(trainer)
    assert params["device"] == "cpu"
    assert params["tree_method"] == "hist"
    assert params["enable_categorical"] is True
    assert params["learning_rate"] == 0.3
    assert params["max_depth"] == 3
    assert params["n_estimators"] == 8
    assert params["eval_metric"] == "auc"


def test_make_trainer_model_kwargs_override_top_level() -> None:
    """``model_kwargs`` wins on key collisions so TOML can override defaults."""
    cfg = TrainingConfig(
        device="cpu",
        metric="auc",
        n_estimators=5,
        max_depth=2,
        early_stopping_rounds=None,
        model_kwargs={"max_depth": 7, "reg_alpha": 0.5},
    )
    trainer = make_trainer(cfg)
    params = _params(trainer)
    assert params["max_depth"] == 7  # overridden
    assert params["reg_alpha"] == 0.5  # added
    assert params["n_estimators"] == 5  # untouched


def test_make_trainer_rejects_non_xgboost_kind() -> None:
    cfg = TrainingConfig(kind="lightgbm", device="cpu", metric="rmse", early_stopping_rounds=None)
    with pytest.raises(NotImplementedError, match="lightgbm"):
        make_trainer(cfg)


def test_trainer_protocol_conforms_to_xgboost(
    training_cfg_classification: TrainingConfig,
) -> None:
    """``XGBClassifier`` satisfies the structural :class:`Trainer` protocol.

    Type-side: ``_accept(trainer)`` is statically checked against the Protocol.
    Runtime-side: every method named in the Protocol exists on the estimator;
    ``best_iteration_`` is only populated by XGBoost when early stopping is
    configured (verified below with an explicit ``early_stopping_rounds`` fit).
    """
    trainer = make_trainer(training_cfg_classification)
    _accept(trainer)
    for attr in ("fit", "predict", "predict_proba"):
        assert callable(getattr(trainer, attr, None)), f"missing {attr}"


def test_trainer_exposes_best_iteration_after_early_stopping_fit() -> None:
    """When ``early_stopping_rounds`` + ``eval_set`` are set, XGBoost populates
    ``best_iteration`` on the fitted estimator (the field the ``Trainer`` Protocol
    advertises as ``best_iteration_`` / sklearn convention).
    """
    cfg = TrainingConfig(
        device="cpu",
        metric="logloss",
        n_estimators=16,
        max_depth=3,
        learning_rate=0.3,
        early_stopping_rounds=2,
    )
    trainer = make_trainer(cfg)
    x = np.random.default_rng(0).normal(size=(40, 3))
    y = (np.random.default_rng(1).random(40) > 0.5).astype(int)
    trainer.fit(x, y, eval_set=[(x, y)], verbose=False)
    best = getattr(trainer, "best_iteration", None)
    assert isinstance(best, int)
