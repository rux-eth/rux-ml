"""PR-054 (program PR-027 A1): the M9 fill fit from per-day batches through the native API.

The batch path builds XGBoost's QuantileDMatrix from a ``DataIter`` that hands over one
day file's partition rows at a time (projection + the row filter + float32), never a
whole-partition frame. Its predictions must be **bit-identical** to the in-memory path
rux-ml ran before (the sklearn wrapper fitted on the whole pandas partitions) on the same
rows: the same QuantileDMatrix cuts, the same booster parameters, the same
early-stopping iteration. XGBoost casts float64 input to float32 itself, so the upstream
cast is value-neutral (program PR-027 R2-b', measured on the real set).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest

from rux_ml._internal.seeds import make_seed_bag
from rux_ml.config import RuxMLConfig
from rux_ml.data import load_parquet, make_splits, materialize
from rux_ml.data.partitions import FOLDS, SourceBatches, plan_partitions
from rux_ml.features import cardinalities_from, make_features
from rux_ml.training import compute_score, make_trainer
from rux_ml.training.xgboost import batches
from rux_ml.training.xgboost.batches import category_levels, fit_batches, predict_fold
from tests.cli.conftest import REPO

STUDIES = {
    "time_ordered": "m9_regime_time_block",
    "symbol_holdout": "m9_regime_symbol_holdout",
    "random": "m9_regime_row_random",
}
TARGET = "y__fill_frac"
TEST = FOLDS.index("test")
VAL = FOLDS.index("val")


def _cfg(c6_set: Path, regime: str, **overrides: object) -> RuxMLConfig:
    sets: dict[str, object] = {
        "data.source_path": str(c6_set / "fill"),
        "training.device": "cpu",
        # subsampling engages the booster's seed and the row order: a stricter identity
        "training.subsample": 0.8,
        "training.colsample_bytree": 0.8,
        **overrides,
    }
    return RuxMLConfig.from_layers(
        REPO / "configs" / "base.toml",
        problem="m9_fill_frac",
        study=STUDIES[regime],
        problems_dir=REPO / "configs" / "problems",
        studies_dir=REPO / "configs" / "studies",
        overrides=sets,
    )


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _in_memory(cfg: RuxMLConfig, split_seed: int, xgb_seed: int) -> dict[str, Any]:
    """The pre-PR-054 ``_fit_and_score`` for an [m9] XGBoost fit: whole frames, sklearn wrapper."""
    assert cfg.data.source_path is not None
    df = materialize(load_parquet(cfg.data.source_path, oracle=cfg.data.oracle))
    splits = make_splits(cfg, df, seed=split_seed)
    x_tr, y_tr = splits["train"].drop(TARGET), splits["train"][TARGET]
    x_va, y_va = splits["val"].drop(TARGET), splits["val"][TARGET]
    cards = cardinalities_from(x_tr, cfg.features.spec.categorical_columns)
    pipeline = make_features(cfg.features, cardinalities=cards)
    pipeline.fit(x_tr, y_tr.to_numpy())  # pyright: ignore[reportUnknownMemberType]

    def pandas(frame: pl.DataFrame) -> pd.DataFrame:
        return cast("pl.DataFrame", pipeline.transform(frame)).to_pandas()  # pyright: ignore[reportUnknownMemberType]

    trainer = cast("Any", make_trainer(cfg.training, seed=xgb_seed))
    x_va_pd = pandas(x_va)
    trainer.fit(pandas(x_tr), y_tr.to_numpy(), eval_set=[(x_va_pd, y_va.to_numpy())], verbose=False)
    return {
        "train_rows": splits["train"].height,
        "best_iteration": trainer.best_iteration,
        "val_pred": trainer.predict(x_va_pd),
        "val_score": compute_score(cfg.training.metric, trainer, x_va_pd, y_va.to_numpy()),
        "test_pred": trainer.predict(pandas(splits["test"].drop(TARGET))),
        "test_realized": splits["test"][TARGET].cast(pl.Float64).fill_nan(None).to_numpy(),
    }


def _batches(cfg: RuxMLConfig, split_seed: int, xgb_seed: int) -> dict[str, Any]:
    assert cfg.data.source_path is not None and cfg.m9 is not None
    source = SourceBatches(
        cfg.data.source_path, oracle=cfg.data.oracle, row_filter=cfg.m9.row_filter_non_null
    )
    plan = plan_partitions(cfg, source, seed=split_seed)
    spec = cfg.features.spec
    cats = category_levels(source, plan, cfg.features)
    fit = fit_batches(cfg, source, plan, TARGET, seed=xgb_seed, categories=cats)
    val_pred, val_y = predict_fold(fit, source, plan, VAL, spec, TARGET, cats)
    test_pred, test_y = predict_fold(fit, source, plan, TEST, spec, TARGET, cats)
    return {
        "train_rows": fit.train_rows,
        "best_iteration": fit.best_iteration,
        "val_pred": val_pred,
        "val_score": compute_score(cfg.training.metric, batches.Scored(), val_pred, val_y),
        "test_pred": test_pred,
        "test_realized": test_y,
        "n_batches": source.n_batches,
    }


@pytest.mark.parametrize("regime", sorted(STUDIES))
def test_the_batch_fit_predicts_bit_identically_to_the_in_memory_fit(
    regime: str, c6_set: Path
) -> None:
    cfg = _cfg(c6_set, regime)
    bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=0)
    ref = _in_memory(cfg, bag.split_seed, bag.xgb_seed)
    got = _batches(cfg, bag.split_seed, bag.xgb_seed)
    assert got["n_batches"] == 3  # three day files: the matrix was built from batches
    assert got["train_rows"] == ref["train_rows"]
    assert got["best_iteration"] == ref["best_iteration"]
    assert _sha(got["val_pred"]) == _sha(ref["val_pred"])
    assert got["val_score"] == ref["val_score"]
    assert _sha(got["test_pred"]) == _sha(ref["test_pred"])
    assert np.array_equal(got["test_realized"], ref["test_realized"], equal_nan=True)


def test_xgboost_is_handed_one_day_of_float32_features_at_a_time(
    c6_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(c6_set, "time_ordered")
    assert cfg.m9 is not None
    seen: list[tuple[int, int, pd.DataFrame, np.ndarray]] = []
    original = batches.fold_batch

    def spy(source: SourceBatches, index: int, plan: Any, fold: int, *a: Any) -> Any:
        x, y = original(source, index, plan, fold, *a)
        seen.append((fold, index, x, y))
        return x, y

    monkeypatch.setattr(batches, "fold_batch", spy)
    bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=0)
    _batches(cfg, bag.split_seed, bag.xgb_seed)
    spec = cfg.features.spec
    train = [(i, x, y) for f, i, x, y in seen if f == FOLDS.index("train")]
    assert len({i for i, _, _ in train}) > 1  # the train matrix came from several days
    for _, i, x, y in seen:
        assert list(x.columns) == [*spec.numeric_columns, *spec.categorical_columns]
        assert all(x[c].dtype == np.float32 for c in spec.numeric_columns)
        assert isinstance(x["kind"].dtype, pd.CategoricalDtype)
        assert not set(x.columns) & {TARGET, *cfg.m9.diagnostic_columns, "stamp_ms", "coin"}
        assert len(x) == len(y) and not np.isnan(y).any()
        day = pl.read_parquet(
            SourceBatches(c6_set / "fill", oracle=cfg.data.oracle, row_filter=[]).files[i]
        )
        assert len(y) < day.height  # a partition's rows of ONE day file, never a whole partition


def test_x_over_the_in_memory_threshold_is_refused_not_silently_streamed(c6_set: Path) -> None:
    cfg = _cfg(c6_set, "time_ordered", **{"data.gpu_in_memory_x_gb_max": 1e-9})
    with pytest.raises(ValueError, match="gpu_in_memory_x_gb_max"):
        _batches(cfg, 1, 2)


def test_a_categorical_that_needs_a_fitted_encoder_is_refused(c6_set: Path) -> None:
    cfg = _cfg(c6_set, "time_ordered", **{"features.categorical_low_card_threshold": 0})
    with pytest.raises(ValueError, match="categorical"):
        _batches(cfg, 1, 2)
