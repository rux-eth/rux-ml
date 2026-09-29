"""The ``[m9]`` fit from per-day batches through XGBoost's native API (PR-054; program PR-027 A1).

At 156 days no route that holds a partition as one pandas frame fits the 20 GiB scope
(program PR-027 R2-b: 52 GiB predicted for per-partition collection; the Polars collect
stays resident, XGBoost adds ≈ 240 B per train row). Here XGBoost builds its quantised
matrix from a :class:`xgboost.DataIter` that hands over **one day file's rows of one
partition at a time** — the key, feature and target columns only, the row filter applied
in the scan, the features cast to float32, the partition rule of
:mod:`rux_ml.data.partitions` applied per row — so the host holds the quantised matrix
plus one day's batch, never a partition's frame. Val is built the same way against the
train cuts (``ref=``); val and test are predicted batch by batch.

**Identity with the in-memory path.** The booster parameters are the sklearn wrapper's
own (``get_xgb_params()`` of the object :func:`make_xgboost_trainer` builds, the one the
in-memory path fits), the matrices take the wrapper's ``missing`` / ``nthread`` /
``max_bin``, the fit is ``xgb.train`` with the wrapper's rounds and early stopping, and
predictions follow ``XGBModel.predict`` (``inplace_predict`` up to the best iteration).
XGBoost stores features as float32 and casts float64 input itself, so the float32 cast
here is value-neutral (program PR-027 R2-b-prime). On the same rows in the same order the
predictions are bit-identical (``tests/cli/test_m9_batches.py``).

**Scope.** An in-memory ``QuantileDMatrix`` only: X above ``data.gpu_in_memory_x_gb_max``
is refused, not silently moved to ``ExtMemQuantileDMatrix`` (not needed at 156 days,
≈ 8.6 GB of float32 X; not built). Regression metrics only (the M9 targets: Brier / MAE).
Categoricals pass through natively only — one fixed category list for every batch — so
a column above ``features.categorical_low_card_threshold`` (a fitted encoder would need
the whole train partition) is refused.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import xgboost as xgb

from rux_ml.data.partitions import TRAIN, VAL
from rux_ml.training.metrics import task_for_metric
from rux_ml.training.xgboost.config import XGBoostTraining
from rux_ml.training.xgboost.factory import make_xgboost_trainer
from rux_ml.training.xgboost.ingest import DEFAULT_BYTES_PER_GB, select_ingest

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from numpy.typing import NDArray

    from rux_ml.config import FeaturesConfig, FeaturesSpec, RuxMLConfig
    from rux_ml.data.partitions import Batches, PartitionPlan

_FEATURE_BYTES = 4  # float32 per value, as XGBoost stores it

# PR-055: what ``predict_fold`` hands a sink per batch — the key columns, the predictions
# and the realized targets of one day file's rows of the partition.
RowSink = Callable[[pl.DataFrame, "NDArray[np.float32]", "NDArray[np.float64]"], None]


def uses_batch_fit(cfg: RuxMLConfig) -> bool:
    """Whether a fit takes the per-day batch path: an ``[m9]`` XGBoost fit of a regression
    target (every M9 target: Brier / MAE). ``rux-ml train`` (PR-054) and the ``promote``
    re-fit (PR-055) both ask this, so a model is re-fitted on the path it was trained on."""
    return (
        cfg.m9 is not None
        and isinstance(cfg.training, XGBoostTraining)
        and task_for_metric(cfg.training.metric) == "regression"
    )


def _dedupe(columns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def category_levels(
    batches: Batches, plan: PartitionPlan, features: FeaturesConfig
) -> dict[str, list[str]]:
    """One category list per categorical column (the sorted values of the whole
    filtered source), so every batch encodes a value with the same code. Refuses a
    column whose train cardinality (nulls counted, as ``cardinalities_from``) is above
    ``features.categorical_low_card_threshold``: the batch fit has no fitted encoder."""
    columns = features.spec.categorical_columns
    if not columns:
        return {}
    threshold = features.categorical_low_card_threshold
    if threshold is None:
        msg = "features.categorical_low_card_threshold is required with categorical columns (D4)"
        raise ValueError(msg)
    levels: dict[str, set[str | None]] = {c: set() for c in columns}
    train: dict[str, set[str | None]] = {c: set() for c in columns}
    for i in range(batches.n_batches):
        frame = batches.frame(i, _dedupe([*plan.columns, *columns]))
        in_train = pl.Series(plan.assign(frame) == TRAIN)
        for c in columns:
            text = frame[c].cast(pl.String)
            levels[c].update(text.unique().to_list())
            train[c].update(text.filter(in_train).unique().to_list())
    for c in columns:
        if len(train[c]) > threshold:
            msg = (
                f"categorical column {c!r} has {len(train[c])} train categories > "
                f"features.categorical_low_card_threshold {threshold}: the per-day batch fit "
                "passes categoricals to XGBoost natively only (an encoder would be fitted on "
                "the whole train partition)"
            )
            raise ValueError(msg)
    return {c: sorted(v for v in levels[c] if v is not None) for c in columns}


def fold_batch(
    batches: Batches,
    index: int,
    plan: PartitionPlan,
    fold: int,
    spec: FeaturesSpec,
    target: str,
    categories: dict[str, list[str]],
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Batch ``index``'s rows of partition ``fold``: X as pandas (the spec's numeric
    columns as float32, then its categoricals on the fixed category list) and the target
    as float64 (null / NaN → NaN)."""
    frame = fold_rows(
        batches, index, plan, fold, [*spec.numeric_columns, *spec.categorical_columns, target]
    )
    return _xy(frame, spec, target, categories)


def fold_rows(
    batches: Batches, index: int, plan: PartitionPlan, fold: int, columns: Sequence[str]
) -> pl.DataFrame:
    """Batch ``index``'s rows of partition ``fold`` (the plan's columns and ``columns``)."""
    frame = batches.frame(index, _dedupe([*plan.columns, *columns]))
    return frame.filter(pl.Series(plan.assign(frame) == fold))


def _xy(
    frame: pl.DataFrame, spec: FeaturesSpec, target: str, categories: dict[str, list[str]]
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    x = frame.select(
        [pl.col(c).cast(pl.Float32) for c in spec.numeric_columns]
        + [pl.col(c).cast(pl.String).cast(pl.Enum(categories[c])) for c in spec.categorical_columns]
    ).to_pandas()
    return x, frame[target].cast(pl.Float64).to_numpy()


class FoldIter(xgb.DataIter):
    """Hands XGBoost the partition's rows one batch (day file) at a time; batches
    without a row of the partition are skipped. ``fold_batch`` is looked up per call."""

    def __init__(
        self,
        batches: Batches,
        plan: PartitionPlan,
        fold: int,
        spec: FeaturesSpec,
        target: str,
        categories: dict[str, list[str]],
    ) -> None:
        super().__init__(release_data=True)
        self._batches = batches
        self._args = (plan, fold, spec, target, categories)
        self._indices = [i for i, rows in enumerate(plan.batch_rows) if rows[fold]]
        self._pos = 0

    def reset(self) -> None:
        self._pos = 0

    def next(self, input_data: Callable[..., None]) -> bool:
        if self._pos >= len(self._indices):
            return False
        x, y = fold_batch(self._batches, self._indices[self._pos], *self._args)
        self._pos += 1
        input_data(data=x, label=y)
        return True


@dataclass
class BatchFit:
    """A fitted booster and what predicting with it as ``XGBModel.predict`` needs."""

    booster: xgb.Booster
    iteration_range: tuple[int, int]
    best_iteration: int | None
    train_rows: int
    missing: float
    nthread: int | None
    enable_categorical: bool
    seconds: dict[str, float] = field(default_factory=dict)


def fit_batches(
    cfg: RuxMLConfig,
    batches: Batches,
    plan: PartitionPlan,
    target: str,
    *,
    seed: int | None,
    categories: dict[str, list[str]],
) -> BatchFit:
    """Build the train / val ``QuantileDMatrix`` from per-day batches and ``xgb.train``."""
    training = cfg.training
    if not isinstance(training, XGBoostTraining):
        msg = f"the per-day batch fit is XGBoost's native API; training.kind is {training.kind!r}"
        raise TypeError(msg)
    if task_for_metric(training.metric) != "regression":
        msg = f"the per-day batch fit trains regression targets; {training.metric!r} is not one"
        raise ValueError(msg)
    spec = cfg.features.spec
    train_rows, val_rows = plan.rows(TRAIN), plan.rows(VAL)
    if not train_rows or not val_rows:
        msg = f"empty partition: train {train_rows} rows, val {val_rows} rows"
        raise ValueError(msg)
    x_bytes = (
        train_rows * (len(spec.numeric_columns) + len(spec.categorical_columns)) * _FEATURE_BYTES
    )
    if select_ingest(x_bytes, cfg.data) is not xgb.QuantileDMatrix:
        msg = (
            f"X train ≈ {x_bytes / DEFAULT_BYTES_PER_GB:.2f} GiB exceeds "
            f"data.gpu_in_memory_x_gb_max {cfg.data.gpu_in_memory_x_gb_max}: the per-day batch "
            "fit builds an in-memory QuantileDMatrix (an ExtMemQuantileDMatrix is not built)"
        )
        raise ValueError(msg)
    model = cast("Any", make_xgboost_trainer(training, seed=seed))
    params = {k: v for k, v in model.get_xgb_params().items() if v is not None}
    matrix: dict[str, Any] = {
        "missing": model.missing,
        "nthread": model.n_jobs,
        "max_bin": model.max_bin,
        "enable_categorical": training.enable_categorical,
    }
    seconds: dict[str, float] = {}
    start = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(FoldIter(batches, plan, TRAIN, spec, target, categories), **matrix)
    seconds["train_matrix"] = time.perf_counter() - start
    start = time.perf_counter()
    dval = xgb.QuantileDMatrix(
        FoldIter(batches, plan, VAL, spec, target, categories), ref=dtrain, **matrix
    )
    seconds["val_matrix"] = time.perf_counter() - start
    start = time.perf_counter()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=model.get_num_boosting_rounds(),
        evals=[(dval, "validation_0")],
        early_stopping_rounds=model.early_stopping_rounds,
        verbose_eval=False,
    )
    seconds["boost"] = time.perf_counter() - start
    del dtrain, dval
    try:  # XGBModel._get_iteration_range: the best iteration when early stopping ran
        best: int | None = int(booster.best_iteration)
        iteration_range = (0, best + 1)
    except AttributeError:
        best, iteration_range = None, (0, 0)
    return BatchFit(
        booster=booster,
        iteration_range=iteration_range,
        best_iteration=best,
        train_rows=train_rows,
        missing=model.missing,
        nthread=model.n_jobs,
        enable_categorical=training.enable_categorical,
        seconds=seconds,
    )


def _predict(fit: BatchFit, x: pd.DataFrame) -> NDArray[np.float32]:
    """``XGBModel.predict``'s route (xgboost 3.2): ``inplace_predict`` over the best
    iteration's range, a ``DMatrix`` when the input type is not supported in place."""
    try:
        out = fit.booster.inplace_predict(
            x,
            iteration_range=fit.iteration_range,
            predict_type="value",
            missing=fit.missing,
            validate_features=True,
        )
    except TypeError:
        dmat = xgb.DMatrix(
            x, missing=fit.missing, nthread=fit.nthread, enable_categorical=fit.enable_categorical
        )
        out = fit.booster.predict(dmat, iteration_range=fit.iteration_range, validate_features=True)
    return np.asarray(out)


def predict_fold(
    fit: BatchFit,
    batches: Batches,
    plan: PartitionPlan,
    fold: int,
    spec: FeaturesSpec,
    target: str,
    categories: dict[str, list[str]],
    *,
    keys: Sequence[str] = (),
    sink: RowSink | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Predictions and realized targets for partition ``fold``, batch by batch, in the
    batches' row order. With ``sink`` (PR-055, the out-of-sample export) each batch's
    ``keys`` columns, predictions and realized targets are handed to it as they are made,
    so the rows are written without ever being held whole."""
    preds: list[NDArray[np.float32]] = []
    realized: list[NDArray[np.float64]] = []
    for i, rows in enumerate(plan.batch_rows):
        if rows[fold]:
            if sink is None:
                x, y = fold_batch(batches, i, plan, fold, spec, target, categories)
                pred = _predict(fit, x)
            else:
                read = [*spec.numeric_columns, *spec.categorical_columns, target, *keys]
                frame = fold_rows(batches, i, plan, fold, read)
                x, y = _xy(frame, spec, target, categories)
                pred = _predict(fit, x)
                sink(frame.select(keys), pred, y)
            preds.append(pred)
            realized.append(y)
    if not preds:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float64)
    return np.concatenate(preds), np.concatenate(realized)


class Scored:
    """A ``Trainer`` whose predictions are already made — the metric registry scores
    trainers, so ``compute_score(metric, Scored(), predictions, y)`` scores them."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Scored:
        msg = "Scored holds predictions; it does not fit"
        raise NotImplementedError(msg)

    def predict(self, X: Any) -> NDArray[Any]:
        return np.asarray(X)

    def predict_proba(self, X: Any) -> NDArray[Any]:
        p = np.asarray(X, dtype=float)
        return np.column_stack([1.0 - p, p]) if p.ndim == 1 else p


def booster_threads_device(booster: xgb.Booster) -> dict[str, int | str]:
    """What the booster ran with (program PR-027 A11), from its saved config:
    ``nthread`` (0 = the OpenMP default) and ``device``."""
    generic = json.loads(booster.save_config())["learner"]["generic_param"]
    return {"nthread": int(generic["nthread"]), "device": str(generic["device"])}
