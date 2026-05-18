"""CPU bit-exact determinism contract (per PR-013 + D9).

XGBoost CPU ``tree_method="hist"`` with single-threaded OpenMP and a pinned
``random_state`` produces bit-exact predictions across runs. This test is
the determinism floor PR-014's golden-regression infrastructure relies on
and is the only place where ``assert_array_equal`` (vs ``assert_allclose``)
is appropriate. GPU near-determinism (atol=1e-5) lives in the gated GPU
test next door.
"""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_classification

from rux_ml._internal.seeds import make_seed_bag
from rux_ml.training import XGBoostTraining, make_trainer


@pytest.fixture(autouse=True)
def _pin_omp_single_thread(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU bit-exact requires single-threaded OpenMP — pin for this module.

    basedpyright doesn't track ``@pytest.fixture(autouse=True)`` usage, so
    the function appears unused; the inline pyright ignore on the def line
    suppresses that.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")


def _synth_dataset() -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Tiny separable binary classification dataset; same shape across runs."""
    x_arr, y_arr = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=0,
    )
    return cast("NDArray[np.float64]", x_arr), cast("NDArray[np.int_]", y_arr)


def _cpu_cfg() -> XGBoostTraining:
    """CPU + hist + early_stopping disabled (val-driven early stopping adds nondet).

    ``subsample`` and ``colsample_bytree`` are set < 1.0 so XGBoost actually
    exercises ``random_state``; with both at 1.0 the booster is deterministic
    regardless of seed (no random draw) and the seed-sensitivity sanity test
    can't distinguish "seed plumbed correctly" from "no randomness present".
    """
    return XGBoostTraining(
        device="cpu",
        tree_method="hist",
        metric="auc",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=None,
        model_kwargs={"n_jobs": 1},  # belt-and-suspenders with OMP_NUM_THREADS=1
    )


def test_cpu_bit_exact_with_pinned_xgb_seed() -> None:
    """Two fits with the same SeedBag → identical predict_proba on the same input."""
    assert os.environ.get("OMP_NUM_THREADS") == "1"  # autouse fixture sanity

    x, y = _synth_dataset()
    bag = make_seed_bag(master_entropy=42, trial_number=0)
    cfg = _cpu_cfg()

    trainer_a = make_trainer(cfg, seed=bag.xgb_seed)
    trainer_a.fit(x, y, verbose=False)
    preds_a = cast(
        "NDArray[np.float64]",
        trainer_a.predict_proba(x),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )

    trainer_b = make_trainer(cfg, seed=bag.xgb_seed)
    trainer_b.fit(x, y, verbose=False)
    preds_b = cast(
        "NDArray[np.float64]",
        trainer_b.predict_proba(x),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )

    np.testing.assert_array_equal(preds_a, preds_b)


def test_cpu_distinct_seeds_produce_distinct_predictions() -> None:
    """Sanity gate: changing ``xgb_seed`` actually shifts predictions.

    Without this, the bit-exact test above could pass spuriously (e.g., if
    XGBoost ignored ``random_state`` entirely for the configured surface).
    """
    x, y = _synth_dataset()
    cfg = _cpu_cfg()

    # Use bags from distinct trial numbers so xgb_seed actually differs.
    bag_a = make_seed_bag(master_entropy=42, trial_number=0)
    bag_b = make_seed_bag(master_entropy=42, trial_number=1)
    assert bag_a.xgb_seed != bag_b.xgb_seed

    trainer_a = make_trainer(cfg, seed=bag_a.xgb_seed)
    trainer_a.fit(x, y, verbose=False)
    preds_a = cast(
        "NDArray[np.float64]",
        trainer_a.predict_proba(x),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )

    trainer_b = make_trainer(cfg, seed=bag_b.xgb_seed)
    trainer_b.fit(x, y, verbose=False)
    preds_b = cast(
        "NDArray[np.float64]",
        trainer_b.predict_proba(x),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )

    # The two should differ on at least one row — subsample/colsample
    # are 1.0 by default so the only randomness is XGBoost's internal RNG;
    # any nonzero difference suffices.
    assert not np.array_equal(preds_a, preds_b)
