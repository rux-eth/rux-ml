"""PR-062: the CPU thread budget (``memory.omp_threads``) reaches the fit as an explicit
argument (program PR-082 R3-a; option A1).

``pin_threads`` exports ``OMP_NUM_THREADS`` after xgboost has loaded, so libgomp's pool
keeps the machine's core count (measured on the desktop: 24 set, 32 effective). These
tests read what the booster was given — ``get_params()`` and the booster's saved config
(``booster_threads_device``) — never the environment (the PR-011 tests' blind spot).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from rux_ml.config import DataConfig
from rux_ml.config.memory import MemoryConfig
from rux_ml.training import (
    CatBoostTraining,
    LightGBMTraining,
    XGBoostNativeAdapter,
    XGBoostTraining,
    make_trainer,
    make_xgboost_trainer,
)
from rux_ml.training.xgboost.batches import booster_threads_device

BUDGET = MemoryConfig(omp_threads=7)  # not 24, not the core count: the value is traced


def _xy(n: int = 120) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    return pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist()}), y


def _xgb(**model_kwargs: object) -> XGBoostTraining:
    return XGBoostTraining(
        device="cpu",
        metric="logloss",
        n_estimators=4,
        max_depth=2,
        learning_rate=0.3,
        early_stopping_rounds=None,
        model_kwargs=model_kwargs,
    )


def _fitted_nthread(trainer: Any) -> int:
    x, y = _xy()
    trainer.fit(x.to_pandas(), y, verbose=False)
    return int(booster_threads_device(trainer.get_booster())["nthread"])


def test_the_factory_takes_the_budget_and_the_booster_runs_with_it() -> None:
    """Test (1): the explicit argument lands in ``n_jobs`` and in the fitted booster."""
    trainer = cast("Any", make_xgboost_trainer(_xgb(), threads=BUDGET.omp_threads))
    assert trainer.get_params()["n_jobs"] == 7
    assert _fitted_nthread(trainer) == 7


def test_make_trainer_passes_the_budget_through() -> None:
    trainer = cast("Any", make_trainer(_xgb(), seed=3, threads=BUDGET.omp_threads))
    assert trainer.get_params()["n_jobs"] == 7
    assert trainer.get_params()["random_state"] == 3


def test_no_budget_leaves_n_jobs_unset() -> None:
    """A caller that passes no budget gets the library default, as before PR-062."""
    trainer = cast("Any", make_xgboost_trainer(_xgb()))
    assert trainer.get_params()["n_jobs"] is None


def test_model_kwargs_still_win_over_the_budget() -> None:
    """Test (2): ``model_kwargs`` ``n_jobs`` (the golden / determinism pin) wins."""
    trainer = cast("Any", make_xgboost_trainer(_xgb(n_jobs=1), threads=BUDGET.omp_threads))
    assert trainer.get_params()["n_jobs"] == 1
    assert _fitted_nthread(trainer) == 1


def _adapter(cfg: XGBoostTraining) -> XGBoostNativeAdapter:
    return XGBoostNativeAdapter(cfg, data_cfg=DataConfig(), memory_cfg=BUDGET, target_column="y")


def test_the_native_adapter_takes_the_budget_from_memory_cfg() -> None:
    """Test (3): ``params['nthread']`` from ``memory_cfg.omp_threads``; the booster has it."""
    adapter = _adapter(_xgb())
    assert adapter._booster_params()["nthread"] == 7  # pyright: ignore[reportPrivateUsage]
    x, y = _xy()
    adapter.fit(x, y)
    assert adapter.booster is not None
    assert booster_threads_device(adapter.booster)["nthread"] == 7


def test_the_native_adapter_lets_model_kwargs_win() -> None:
    adapter = _adapter(_xgb(nthread=2))
    assert adapter._booster_params()["nthread"] == 2  # pyright: ignore[reportPrivateUsage]
    x, y = _xy()
    adapter.fit(x, y)
    assert adapter.booster is not None
    assert booster_threads_device(adapter.booster)["nthread"] == 2


def _catboost() -> CatBoostTraining:
    return CatBoostTraining(
        device="cpu", metric="auc", iterations=4, depth=2, learning_rate=0.3,
        early_stopping_rounds=None,
    )  # fmt: skip


def test_catboost_takes_the_same_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate lean 5: one transport. The argument sets ``thread_count``; the environment no
    longer does (it was the second transport, and ``pin_threads`` set it only per verb)."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    trainer = cast("Any", make_trainer(_catboost(), threads=BUDGET.omp_threads))
    assert trainer.get_params()["thread_count"] == 7
    unset = cast("Any", make_trainer(_catboost()))
    # -1 = CatBoost's own default (all cores); its get_params leaves the default out
    assert unset.get_params().get("thread_count", -1) == -1


def test_lightgbm_accepts_the_argument_and_ignores_it() -> None:
    cfg = LightGBMTraining(device="cpu", metric="auc", n_estimators=4, max_depth=2)
    with_budget = cast("Any", make_trainer(cfg, threads=BUDGET.omp_threads))
    without = cast("Any", make_trainer(cfg))
    assert with_budget.get_params() == without.get_params()
