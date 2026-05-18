"""End-to-end smoke for the CatBoost Trainer family (per PR-019).

Three surfaces beyond the parametrized conformance test in
``test_registry_conformance.py``:

1. **`cat_features` auto-extraction at fit time** — verifies the Q-Cat finding
   that the ``_CatBoostTrainerShim`` extracts categorical-dtype column names
   from the input DataFrame and passes them through to ``CatBoostClassifier.fit``.
   The synthetic dataset has a 3-level pandas Categorical column, which would
   trigger CatBoost's "dtype 'category' but not in cat_features list" error
   if the shim weren't catching it.
2. **`early_stopping_rounds` constructor kwarg** — verifies that CatBoost's
   simpler integration (vs LightGBM's fit-time callback) works through the
   shim unchanged.
3. **Metric translation** — Q-Wrap finding: workbench ``auc`` →
   CatBoost ``AUC``.

GPU support is the default per Q-GPU, but the GPU smoke test is gated by
``@pytest.mark.gpu`` so CPU-only CI can opt out.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from rux_ml.training import CatBoostTraining, make_trainer


def _build_dataset_with_categorical() -> tuple[pd.DataFrame, NDArray[np.int_]]:
    """100 rows, 3 numeric + 1 low-card categorical (3 levels) + binary target.

    Mirrors ``tests/training/test_lightgbm_smoke.py`` shape so the two
    family smoke tests are directly comparable.
    """
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = ((0.5 * x1 + 0.3 * x2 + 0.2 * x3) > 0).astype(int)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "cat": pd.Categorical(cat, categories=["a", "b", "c"]),
        }
    )
    return df, cast("NDArray[np.int_]", y)


def test_catboost_fit_predict_with_categorical_column() -> None:
    """The shim extracts cat_features from pandas Categorical-dtype columns (Q-Cat)."""
    cfg = CatBoostTraining(
        device="cpu",
        metric="auc",
        iterations=8,
        depth=3,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )
    trainer = make_trainer(cfg)

    x, y = _build_dataset_with_categorical()
    fitted = trainer.fit(x, y)
    assert fitted is trainer  # sklearn fit-returns-self (preserved through the shim)

    preds = np.asarray(trainer.predict(x))
    assert preds.shape == (100,)
    assert set(int(p) for p in preds) <= {0, 1}


def test_catboost_early_stopping_constructor_kwarg() -> None:
    """``early_stopping_rounds`` is a CatBoost constructor kwarg (Q-Wrap §2).

    Unlike LightGBM (fit-time callback), CatBoost takes it directly at
    construction. The factory passes it via ``_catboost_kwargs``. Test:
    train with aggressive early-stopping + matched eval_set; verify that
    ``best_iteration_`` is populated as an integer (callback fired without
    error).
    """
    cfg = CatBoostTraining(
        device="cpu",
        metric="logloss",
        iterations=200,
        depth=3,
        learning_rate=0.3,
        early_stopping_rounds=5,
    )
    trainer = make_trainer(cfg)

    x, y = _build_dataset_with_categorical()
    trainer.fit(x, y, eval_set=(x, y))

    best_iter = getattr(trainer, "best_iteration_", None)
    assert isinstance(best_iter, int), f"best_iteration_ not populated: {best_iter!r}"


def test_catboost_auc_metric_translated_to_capitalized_AUC() -> None:
    """Q-Wrap: workbench ``auc`` -> CatBoost ``AUC`` (case-sensitive)."""
    cfg = CatBoostTraining(
        device="cpu",
        metric="auc",
        iterations=4,
        depth=2,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )
    trainer = make_trainer(cfg)
    # Shim forwards get_params() to the underlying CatBoostClassifier.
    params = trainer.get_params()  # pyright: ignore[reportAttributeAccessIssue]
    # CatBoost's eval_metric is set to the translated name.
    assert params["eval_metric"] == "AUC"
    # loss_function is task-derived (classifier -> Logloss).
    assert params["loss_function"] == "Logloss"


def test_catboost_logloss_metric_translated_to_capitalized_Logloss() -> None:
    """Q-Wrap: workbench ``logloss`` -> CatBoost ``Logloss``."""
    cfg = CatBoostTraining(
        device="cpu",
        metric="logloss",
        iterations=4,
        depth=2,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )
    trainer = make_trainer(cfg)
    params = trainer.get_params()  # pyright: ignore[reportAttributeAccessIssue]
    assert params["eval_metric"] == "Logloss"
    assert params["loss_function"] == "Logloss"


def test_catboost_rmse_metric_uses_regressor() -> None:
    """Task derivation: ``rmse`` -> CatBoostRegressor with ``loss_function=RMSE``."""
    from catboost import CatBoostRegressor  # noqa: PLC0415

    cfg = CatBoostTraining(
        device="cpu",
        metric="rmse",
        iterations=4,
        depth=2,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )
    trainer = make_trainer(cfg)
    assert isinstance(trainer._base, CatBoostRegressor)  # pyright: ignore[reportAttributeAccessIssue]
    params = trainer.get_params()  # pyright: ignore[reportAttributeAccessIssue]
    assert params["eval_metric"] == "RMSE"
    assert params["loss_function"] == "RMSE"


def test_catboost_task_type_gpu_when_device_cuda() -> None:
    """Q-GPU: ``cfg.device == "cuda"`` activates ``task_type="GPU"`` + ``devices="0"``."""
    cfg = CatBoostTraining(
        device="cuda",
        metric="auc",
        iterations=4,
        depth=2,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )
    trainer = make_trainer(cfg)
    params = trainer.get_params()  # pyright: ignore[reportAttributeAccessIssue]
    assert params["task_type"] == "GPU"
    assert params["devices"] == "0"


def test_catboost_task_type_cpu_when_device_cpu() -> None:
    """``cfg.device == "cpu"`` activates ``task_type="CPU"``; no ``devices`` kwarg."""
    cfg = CatBoostTraining(
        device="cpu",
        metric="auc",
        iterations=4,
        depth=2,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )
    trainer = make_trainer(cfg)
    params = trainer.get_params()  # pyright: ignore[reportAttributeAccessIssue]
    assert params["task_type"] == "CPU"
    # ``devices`` is only set when GPU.
    assert "devices" not in params or params.get("devices") in (None, "")


def test_catboost_thread_count_from_omp_num_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-Parallel: factory reads OMP_NUM_THREADS and passes as ``thread_count``."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    cfg = CatBoostTraining(
        device="cpu",
        metric="auc",
        iterations=4,
        depth=2,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )
    trainer = make_trainer(cfg)
    params = trainer.get_params()  # pyright: ignore[reportAttributeAccessIssue]
    assert params["thread_count"] == 3
