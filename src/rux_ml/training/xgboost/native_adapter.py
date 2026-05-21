"""``XGBoostNativeAdapter`` — Trainer-Protocol-compatible native ``xgb.train`` adapter (per PR-033).

Activates the ExtMem out-of-core ingest path the workbench has carried in
design since D3/PR-006 but never actually exercised end-to-end in production.
Parallel to PR-032's score-only ``_BoosterTrainerShim`` (which wraps a
bundle-loaded ``Booster`` for inference); this adapter owns the *training*
side, calling ``xgb.train`` directly so the ingest decision from
``select_ingest`` (``QuantileDMatrix`` vs ``ExtMemQuantileDMatrix``) drives
actual DMatrix construction.

Activation: opt-in via ``cfg.training.use_native=True`` only. The XGBoost 3.2
external-memory tutorial warns that ``ExtMemQuantileDMatrix`` is slower than
``QuantileDMatrix`` when data fits in host RAM — auto-on-ExtMem-trigger is
deferred to a follow-up once a real workload exceeds the 18 GB threshold.
HPO objective stays UNTOUCHED — per-fold ExtMem rebuild cost is prohibitive
and ``_check_extmem_compat`` preserves the "HPO + ExtMem = NotImplementedError"
semantic.

Parameter mapping (XGBoost 3.2 ``python_api.html``): sklearn-wrapper
``random_state``/``n_estimators`` become native ``seed``/``num_boost_round``;
``enable_categorical`` + ``cache_host_ratio`` are DMatrix construction
kwargs, not booster params; ``early_stopping_rounds`` is an ``xgb.train``
kwarg.

**Status**: ``best-guess-given-constraints`` for the 4-element combination
(``xgb.train`` + ``ExtMemQuantileDMatrix`` + ``cache_host_ratio`` +
sklearn-Trainer-Protocol wrapper) — no single cited working example was
located at Phase 3. Mitigation: a mandatory numeric-agreement test
(``test_native_adapter_predict_proba_matches_sklearn_wrapper_within_tol``)
asserts ``predict_proba`` matches the sklearn-wrapper path within ``atol=1e-5``
on shared params + seed + data; without it the BGGC label would be unbounded.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import xgboost as xgb

from rux_ml.data.data_iter import single_source_iter
from rux_ml.training.metrics import task_for_metric
from rux_ml.training.xgboost.ingest import estimate_x_bytes, select_ingest

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rux_ml.config.data import DataConfig
    from rux_ml.config.memory import MemoryConfig
    from rux_ml.training.xgboost.config import XGBoostTraining


# Number of ExtMem batches when chunking a single-source frame. Small enough
# that test-mode forced-ExtMem paths produce >1 batch (exercising the
# iterator's reset/next contract), large enough to stay near NVIDIA's
# "5-10 GB per batch" recommendation on real 36 GB-host workloads.
_DEFAULT_EXTMEM_BATCH_COUNT: int = 4


class XGBoostNativeAdapter:
    """Native ``xgb.train`` adapter exposing the sklearn ``Trainer`` Protocol.

    ``fit(X, y, ..., eval_set=...)`` branches on
    :func:`rux_ml.training.xgboost.ingest.select_ingest`: in-VRAM data builds
    a :class:`xgb.QuantileDMatrix` directly from the frames; over-threshold
    data writes ``_DEFAULT_EXTMEM_BATCH_COUNT`` temp Parquets via
    :func:`single_source_iter` and wraps them in
    :class:`xgb.ExtMemQuantileDMatrix` with the configured
    ``cache_host_ratio``.

    ``predict`` / ``predict_proba`` mirror PR-032's ``_BoosterTrainerShim``
    (``xgb.DMatrix(x, enable_categorical=True) → booster.predict(...)``;
    ``[1-p, p]`` 2D shape for binary classification) so downstream consumers
    of the metric registry see the same surface as the sklearn-wrapper path.
    """

    def __init__(
        self,
        cfg: XGBoostTraining,
        *,
        data_cfg: DataConfig,
        memory_cfg: MemoryConfig,
        target_column: str,
        seed: int | None = None,
    ) -> None:
        self._cfg = cfg
        self._data_cfg = data_cfg
        self._memory_cfg = memory_cfg
        self._target_column = target_column
        self._seed = seed
        self._booster: xgb.Booster | None = None
        self._task: str = task_for_metric(cfg.metric)

    # ------------------------------------------------------------------
    # Parameter assembly (native xgb.train signature)
    # ------------------------------------------------------------------

    def _booster_params(self) -> dict[str, Any]:
        """Translate ``XGBoostTraining`` fields into ``xgb.train`` ``params`` dict.

        XGBoost 3.2 ``python_api.html`` distinguishes booster ``params``
        (passed inside the dict) from train-loop kwargs (``num_boost_round``,
        ``early_stopping_rounds``, ``evals``) and from DMatrix construction
        kwargs (``enable_categorical``, ``cache_host_ratio``). Only the
        first group goes here.

        Renames from the sklearn wrapper at
        :func:`rux_ml.training.xgboost.factory._xgb_kwargs`:
        ``random_state`` → ``seed``; ``n_estimators`` does NOT belong here —
        it's the ``num_boost_round`` kwarg to :func:`xgb.train`.

        ``model_kwargs`` last-wins matches PR-006's convention so TOML
        overrides apply without per-field plumbing.
        """
        params: dict[str, Any] = {
            "device": self._cfg.device,
            "tree_method": self._cfg.tree_method,
            "learning_rate": self._cfg.learning_rate,
            "max_depth": self._cfg.max_depth,
            "subsample": self._cfg.subsample,
            "colsample_bytree": self._cfg.colsample_bytree,
            "eval_metric": self._cfg.metric,
            "objective": _objective_for_task(self._task),
        }
        if self._seed is not None:
            params["seed"] = self._seed
        params.update(self._cfg.model_kwargs)
        return params

    # ------------------------------------------------------------------
    # DMatrix construction (branches on select_ingest)
    # ------------------------------------------------------------------

    def _build_train_dmatrix(
        self,
        x: pl.DataFrame,
        y: NDArray[Any],
        *,
        tmp_dir: Path,
    ) -> xgb.QuantileDMatrix | xgb.ExtMemQuantileDMatrix:
        dmatrix_cls = select_ingest(estimate_x_bytes(x), self._data_cfg)
        if dmatrix_cls is xgb.QuantileDMatrix:
            return xgb.QuantileDMatrix(
                x.to_pandas(),
                label=y,
                enable_categorical=True,
            )
        # ExtMem branch: chunk to N temp Parquets via single_source_iter.
        # ParquetDataIter pops the target column per batch, so it must live
        # in the chunked frames.
        chunk_frame = x.with_columns(pl.Series(self._target_column, y))
        data_iter = single_source_iter(
            chunk_frame,
            self._target_column,
            batch_count=_DEFAULT_EXTMEM_BATCH_COUNT,
            tmp_dir=tmp_dir,
            cache_prefix=str(tmp_dir / "xgb_cache"),
        )
        kwargs: dict[str, Any] = {"enable_categorical": True}
        # XGBoost 3.2 only honors `cache_host_ratio` on GPU `ExtMemQuantileDMatrix`
        # (CPU build raises "cache_host_ratio is only used by the GPU
        # ExtMemQuantileDMatrix"). Gate on the trainer's device so CPU
        # smoke tests stay clean while the real workbench GPU path still
        # consumes `cfg.memory.cache_host_ratio`.
        if (
            self._memory_cfg.cache_host_ratio is not None
            and self._cfg.device == "cuda"
        ):
            kwargs["cache_host_ratio"] = self._memory_cfg.cache_host_ratio
        return xgb.ExtMemQuantileDMatrix(data_iter, **kwargs)

    # ------------------------------------------------------------------
    # Trainer Protocol surface
    # ------------------------------------------------------------------

    def fit(
        self,
        x: pl.DataFrame,
        y: NDArray[Any] | pl.Series,
        *,
        eval_set: list[tuple[pl.DataFrame, NDArray[Any] | pl.Series]] | None = None,
        verbose: bool = False,
    ) -> XGBoostNativeAdapter:
        """Fit the native booster.

        ``X`` is a Polars DataFrame post feature pipeline (matches PR-006
        ``cli/train._fit_and_score`` call shape after the
        ``cast(pl.DataFrame, pipeline.transform(...))`` step). ``y`` accepts
        numpy or Polars to mirror the existing call sites.
        """
        y_arr = _to_numpy(y)
        eval_pairs = [(pl_x, _to_numpy(pl_y)) for pl_x, pl_y in (eval_set or [])]

        # ``TemporaryDirectory`` cleans up the chunked Parquets AND XGBoost's
        # ExtMem cache files on context exit. Owned per fit call.
        with tempfile.TemporaryDirectory(prefix="rux_ml_extmem_") as owned:
            dtrain = self._build_train_dmatrix(x, y_arr, tmp_dir=Path(owned))
            evals: list[tuple[xgb.DMatrix, str]] = []
            for i, (eval_x, eval_y) in enumerate(eval_pairs):
                # Eval frames are typically the val partition (small); a
                # plain DMatrix is sufficient and avoids chunking them too.
                deval = xgb.DMatrix(
                    eval_x.to_pandas(),
                    label=eval_y,
                    enable_categorical=True,
                )
                evals.append((deval, f"val_{i}"))
            train_kwargs: dict[str, Any] = {
                "params": self._booster_params(),
                "dtrain": dtrain,
                "num_boost_round": self._cfg.n_estimators,
            }
            if evals:
                train_kwargs["evals"] = evals
            if self._cfg.early_stopping_rounds is not None and evals:
                train_kwargs["early_stopping_rounds"] = self._cfg.early_stopping_rounds
            self._booster = xgb.train(**train_kwargs)
        return self

    def predict(self, x: Any) -> NDArray[Any]:
        """Predict from any sklearn-compatible 2D input (pandas / polars / ndarray).

        Mirrors PR-032's ``_BoosterTrainerShim._dmatrix``: pass the input
        straight to :class:`xgb.DMatrix` so callers don't have to convert.
        ``compute_score`` passes pandas; the CLI passes pandas;
        promotion-time consumers may pass polars.
        """
        if self._booster is None:
            msg = "XGBoostNativeAdapter.predict called before fit()"
            raise RuntimeError(msg)
        dmat = xgb.DMatrix(x, enable_categorical=True)
        return np.asarray(self._booster.predict(dmat))

    def predict_proba(self, x: Any) -> NDArray[Any]:
        """Return ``[1-p, p]`` 2D shape for binary classification (mirrors PR-032 shim)."""
        if self._booster is None:
            msg = "XGBoostNativeAdapter.predict_proba called before fit()"
            raise RuntimeError(msg)
        dmat = xgb.DMatrix(x, enable_categorical=True)
        p = np.asarray(self._booster.predict(dmat))
        if p.ndim == 1:
            return np.column_stack([1.0 - p, p])
        return p

    @property
    def best_iteration(self) -> int | None:
        """Native booster's ``best_iteration`` after early stopping (else ``None``)."""
        if self._booster is None:
            return None
        return cast("int | None", getattr(self._booster, "best_iteration", None))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _objective_for_task(task: str) -> str:
    """Map task to native XGBoost objective string.

    sklearn ``XGBClassifier`` defaults to ``binary:logistic`` for binary
    classification and ``reg:squarederror`` for regression; native
    ``xgb.train`` does not auto-detect — the caller must pass it via params.
    Multi-class is out of scope at v0 (no concrete workbench problem yet).
    """
    if task == "classification":
        return "binary:logistic"
    return "reg:squarederror"


def _to_numpy(y: NDArray[Any] | pl.Series) -> NDArray[Any]:
    """Accept numpy or polars label inputs uniformly."""
    if isinstance(y, pl.Series):
        return y.to_numpy()
    return np.asarray(y)
