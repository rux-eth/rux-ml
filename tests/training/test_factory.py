"""Unit tests for ``make_trainer`` — concrete class + kwarg propagation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError
from xgboost import XGBClassifier, XGBModel, XGBRegressor

from rux_ml.training import TRAINER_FAMILIES, Trainer, XGBoostTraining, make_trainer
from rux_ml.training.xgboost.factory import _xgb_kwargs  # pyright: ignore[reportPrivateUsage]


def _accept(_trainer: Trainer) -> None:
    """Static-typing acceptance helper: basedpyright fails if ``XGBClassifier``
    does not conform to :class:`Trainer`. At runtime this is a no-op."""


def test_make_trainer_returns_xgbclassifier_for_classification_metric(
    training_cfg_classification: XGBoostTraining,
) -> None:
    trainer = make_trainer(training_cfg_classification)
    assert isinstance(trainer, XGBClassifier)


def test_make_trainer_returns_xgbregressor_for_regression_metric(
    training_cfg_regression: XGBoostTraining,
) -> None:
    trainer = make_trainer(training_cfg_regression)
    assert isinstance(trainer, XGBRegressor)


def _params(trainer: Trainer) -> dict[str, object]:
    """Narrow the universal ``Trainer`` Protocol to the sklearn-shape ``get_params``."""
    return cast("XGBModel", trainer).get_params()


def test_make_trainer_propagates_top_level_kwargs(
    training_cfg_classification: XGBoostTraining,
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
    cfg = XGBoostTraining(
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


def test_pydantic_rejects_unknown_family_at_config_validation() -> None:
    """Discriminated union: ``kind="lightgbm"`` is rejected at config construction.

    Pre-PR-017 the rejection happened later (``make_trainer`` raised
    ``NotImplementedError``). Post-refactor, Pydantic's discriminated union
    enforces the rejection at validation time — the unknown ``kind`` value
    can never reach the factory dispatcher.
    """
    with pytest.raises(ValidationError):
        # The discriminated union's literal type rejects "lightgbm" until PR-018
        # adds it. Until then, validation fails with a ValidationError naming
        # the invalid literal.
        XGBoostTraining(  # pyright: ignore[reportCallIssue]
            kind="lightgbm",  # type: ignore[arg-type]
            device="cpu",
            metric="rmse",
            early_stopping_rounds=None,
        )


def test_make_trainer_rejects_unregistered_family() -> None:
    """Defensive: ``make_trainer`` raises ValueError when ``cfg.kind`` is not in
    ``TRAINER_FAMILIES``.

    Reachable only via a caller that bypasses Pydantic validation (which is
    blocked by ``test_pydantic_rejects_unknown_family_at_config_validation``),
    or via a programmatic mutation of ``TRAINER_FAMILIES`` itself. Kept as a
    defensive smoke so the error path is exercised in CI and the dispatcher
    can never silently fall through.
    """

    class _ForgedConfig:
        kind = "bogus_family"

    with pytest.raises(ValueError, match="not registered"):
        make_trainer(cast("XGBoostTraining", _ForgedConfig()))


def test_trainer_families_registry_contains_xgboost() -> None:
    """``TRAINER_FAMILIES`` is the SSOT for registered families."""
    assert "xgboost" in TRAINER_FAMILIES
    assert callable(TRAINER_FAMILIES["xgboost"])


def test_trainer_protocol_conforms_to_xgboost(
    training_cfg_classification: XGBoostTraining,
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
    cfg = XGBoostTraining(
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


def test_brier_early_stops_on_xgboost_rmse() -> None:
    """XGBoost has no 'brier' eval metric; Brier = RMSE^2 on a probability, so early
    stopping on rmse is the same ordering (PR-041)."""
    cfg = XGBoostTraining(metric="brier")
    assert _xgb_kwargs(cfg, seed=None)["eval_metric"] == "rmse"
    assert isinstance(make_trainer(cfg), XGBRegressor)


def test_early_stopping_rounds_zero_is_off_at_the_installed_xgboost() -> None:
    """PR-060 (program PR-082 lean 3): ``early_stopping_rounds = 0`` through rux-ml's own
    factory boosts every round although an ``eval_set`` is given — the probe cells' "no
    validation-driven early stopping" rests on it. Behaviour, not a version string: the
    floor ``xgboost>=3.1`` admits versions this gate was not read on. The control (2)
    proves the data does stop a fit early, so the check is not vacuous."""
    import xgboost  # noqa: PLC0415

    rng = np.random.default_rng(0)
    x, x_val = rng.normal(size=(200, 5)), rng.normal(size=(200, 5))
    y, y_val = (rng.random(200) > 0.5).astype(int), (rng.random(200) > 0.5).astype(int)
    n_estimators = 40

    def fit(rounds: int) -> XGBClassifier:
        cfg = XGBoostTraining(
            device="cpu",
            metric="logloss",
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.5,
            early_stopping_rounds=rounds,
        )
        trainer = cast("XGBClassifier", make_trainer(cfg, seed=0))
        trainer.fit(x, y, eval_set=[(x_val, y_val)], verbose=False)
        return trainer

    where = f"xgboost {xgboost.__version__}"
    control = fit(2)  # noise labels: the val loss rises, so 2 stops the fit early
    assert control.get_booster().num_boosted_rounds() < n_estimators, where
    off = fit(0)
    assert off.get_booster().num_boosted_rounds() == n_estimators, where
    assert getattr(off, "best_iteration", None) is None, where
    np.testing.assert_array_equal(
        off.predict_proba(x_val), off.predict_proba(x_val, iteration_range=(0, n_estimators))
    )
