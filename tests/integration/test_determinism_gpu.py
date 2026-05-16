"""GPU near-determinism contract (per PR-013 + D9).

XGBoost ``device="cuda"`` + ``tree_method="hist"`` with a pinned
``random_state`` is **near-deterministic, not bit-exact** across hardware
(per D9 + ``docs/CONSTRAINTS.md`` tolerance-based golden-tests rule). Two
runs on the same hardware with the same seed should match within
``atol=1e-5``.

Gated behind ``@pytest.mark.gpu`` — the default ``uv run pytest`` excludes
``-m gpu``. Run on the Linux workbench via:

    uv run pytest -m gpu tests/integration/test_determinism_gpu.py

If this test starts FAILING intermittently, do NOT loosen the tolerance
silently — investigate the change against XGBoost release notes per the
PR-013 spec's "GPU determinism may regress" guidance.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_classification

from rux_ml._internal.seeds import make_seed_bag
from rux_ml.config import TrainingConfig
from rux_ml.training import make_trainer

pytestmark = pytest.mark.gpu


def _synth_dataset() -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    x_arr, y_arr = make_classification(
        n_samples=512,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=0,
    )
    return cast("NDArray[np.float64]", x_arr), cast("NDArray[np.int_]", y_arr)


def _gpu_cfg() -> TrainingConfig:
    return TrainingConfig(
        device="cuda",
        tree_method="hist",
        metric="auc",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )


def test_gpu_near_deterministic_within_atol_1e_minus_5() -> None:
    """Two fits on the same GPU with the same SeedBag → predictions within atol=1e-5."""
    x, y = _synth_dataset()
    bag = make_seed_bag(master_entropy=42, trial_number=0)
    cfg = _gpu_cfg()

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

    # Per D9 + CONSTRAINTS.md: never `assert_array_equal` on GPU output.
    np.testing.assert_allclose(preds_a, preds_b, atol=1e-5, rtol=1e-4)
