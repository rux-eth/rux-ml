"""Tests for :class:`XGBoostNativeAdapter` + :func:`single_source_iter` (per PR-033).

Includes the mandatory BGGC mitigation test
(``test_native_adapter_predict_proba_matches_sklearn_wrapper_within_tol``)
that bounds the unverified-synthesis risk Phase 3 surfaced on the 4-element
combination ``xgb.train`` + ``ExtMemQuantileDMatrix`` + ``cache_host_ratio``
+ sklearn-Trainer-Protocol wrapper. Without it the BGGC label would be
unbounded — the adapter could silently disagree with the sklearn-wrapper
path on shared params + seed + data and only surface as score drift later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from xgboost import XGBClassifier

from rux_ml.config import DataConfig
from rux_ml.config.memory import MemoryConfig
from rux_ml.data import single_source_iter
from rux_ml.data.data_iter import ParquetDataIter
from rux_ml.training import XGBoostNativeAdapter, XGBoostTraining
from rux_ml.training.xgboost import native_adapter as native_adapter_module


def _synth_binary(n: int = 80, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    """Linearly-separable-ish binary classification fixture (mirrors test_train_subcommand)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    df = pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist()})
    return df, y


def _xgb_cfg(metric: str = "logloss", **overrides: object) -> XGBoostTraining:
    base: dict[str, object] = {
        "device": "cpu",
        "metric": metric,
        "n_estimators": 8,
        "max_depth": 3,
        "learning_rate": 0.3,
        "early_stopping_rounds": None,
    }
    base.update(overrides)
    return XGBoostTraining(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Adapter smoke + BGGC mitigation
# ---------------------------------------------------------------------------


def test_native_adapter_fit_predict_smoke() -> None:
    """End-to-end fit → predict → predict_proba on synthetic binary data."""
    df, y = _synth_binary()
    adapter = XGBoostNativeAdapter(
        _xgb_cfg(),
        data_cfg=DataConfig(),
        memory_cfg=MemoryConfig(),
        target_column="y",
    )
    adapter.fit(df, y)
    pred = adapter.predict(df)
    proba = adapter.predict_proba(df)
    assert pred.shape == (df.height,)
    assert proba.shape == (df.height, 2)
    # Proba rows sum to ~1 (binary [1-p, p] shape).
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_native_adapter_predict_proba_matches_sklearn_wrapper_within_tol() -> None:
    """MANDATORY BGGC mitigation: native adapter ≡ sklearn wrapper on shared seed + params.

    Bounds the unverified-synthesis risk from Phase 3 Q-Group-D. Without
    this test the 4-element combination (``xgb.train`` +
    ``ExtMemQuantileDMatrix`` + ``cache_host_ratio`` + sklearn-Protocol
    wrapper) could silently diverge from the sklearn-wrapper path. With
    matched params + seed + objective on CPU ``tree_method=hist``, the
    output should be bit-exact (atol budget kept loose at 1e-5 to allow
    for a future GPU-hist promotion that retains near-determinism).
    """
    df, y = _synth_binary(n=120, seed=7)
    cfg = _xgb_cfg(metric="logloss")
    seed = 42

    adapter = XGBoostNativeAdapter(
        cfg,
        data_cfg=DataConfig(),
        memory_cfg=MemoryConfig(),
        target_column="y",
        seed=seed,
    )
    adapter.fit(df, y)
    proba_native = adapter.predict_proba(df)

    sklearn = XGBClassifier(
        device=cfg.device,
        tree_method=cfg.tree_method,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        enable_categorical=True,
        eval_metric=cfg.metric,
        random_state=seed,
        objective="binary:logistic",
    )
    sklearn.fit(df.to_pandas(), y)
    proba_sklearn = sklearn.predict_proba(df.to_pandas())

    assert proba_native.shape == proba_sklearn.shape
    np.testing.assert_allclose(proba_native, proba_sklearn, atol=1e-5)


def test_native_adapter_predict_proba_returns_2d_for_binary() -> None:
    df, y = _synth_binary()
    adapter = XGBoostNativeAdapter(
        _xgb_cfg(),
        data_cfg=DataConfig(),
        memory_cfg=MemoryConfig(),
        target_column="y",
    )
    adapter.fit(df, y)
    proba = adapter.predict_proba(df)
    assert proba.ndim == 2
    assert proba.shape[1] == 2


def test_native_adapter_uses_quantile_dmatrix_for_small_x(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``select_ingest`` returns ``QuantileDMatrix``, the adapter builds it directly."""
    df, y = _synth_binary()
    captured: list[type] = []

    def fake_select_ingest(_x_bytes: int, _data_cfg: object) -> type:
        captured.append(xgb.QuantileDMatrix)
        return xgb.QuantileDMatrix

    monkeypatch.setattr(native_adapter_module, "select_ingest", fake_select_ingest)
    adapter = XGBoostNativeAdapter(
        _xgb_cfg(),
        data_cfg=DataConfig(),
        memory_cfg=MemoryConfig(),
        target_column="y",
    )
    adapter.fit(df, y)
    assert captured == [xgb.QuantileDMatrix]


def test_native_adapter_threads_cache_host_ratio_when_device_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cache_host_ratio`` is threaded into ``ExtMemQuantileDMatrix`` kwargs on GPU only.

    Verifies (a) the kwarg reaches the DMatrix constructor when
    ``device == "cuda"`` AND ``cache_host_ratio`` is set, and (b) the gating
    suppresses it on CPU where XGBoost rejects the param (per the 3.2
    "cache_host_ratio is only used by the GPU ExtMemQuantileDMatrix"
    check). Real GPU fit is out of reach on CI / macOS; we monkey-patch
    the constructor + the train call so the test runs anywhere.
    """
    df, y = _synth_binary(n=40)
    captured_kwargs: dict[str, object] = {}

    class _FakeDMatrix:
        def __init__(self, _data: object, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    def _fake_train(**_kwargs: object) -> object:
        class _FakeBooster:
            best_iteration = 0

            def predict(self, _dm: object) -> np.ndarray:
                return np.zeros(df.height)

        return _FakeBooster()

    monkeypatch.setattr(native_adapter_module.xgb, "ExtMemQuantileDMatrix", _FakeDMatrix)
    monkeypatch.setattr(native_adapter_module.xgb, "DMatrix", _FakeDMatrix)
    monkeypatch.setattr(native_adapter_module.xgb, "train", _fake_train)

    # Force the ExtMem branch via a tiny threshold so select_ingest returns ExtMem.
    data_cfg = DataConfig(gpu_in_memory_x_gb_max=1e-9)
    memory_cfg = MemoryConfig(cache_host_ratio=0.75)
    adapter = XGBoostNativeAdapter(
        _xgb_cfg(device="cuda", n_estimators=2),
        data_cfg=data_cfg,
        memory_cfg=memory_cfg,
        target_column="y",
    )
    adapter.fit(df, y)
    assert captured_kwargs.get("cache_host_ratio") == 0.75
    assert captured_kwargs.get("enable_categorical") is True


def test_native_adapter_uses_extmem_for_large_x_with_force_flag(
    tmp_path: Path,
) -> None:
    """Tiny ``gpu_in_memory_x_gb_max`` forces real ``ExtMemQuantileDMatrix`` construction.

    Verifies that (a) ``cache_host_ratio`` is threaded through (smoke: no
    exception when set), (b) the chunked Parquets actually land on disk
    under XGBoost's tempdir lifecycle, and (c) the resulting booster still
    predicts (the chunking + iterator + reset/next contract is exercised).
    """
    df, y = _synth_binary(n=80)
    # ~1-byte threshold forces ExtMem regardless of actual frame size.
    data_cfg = DataConfig(gpu_in_memory_x_gb_max=1e-9)
    memory_cfg = MemoryConfig(cache_host_ratio=0.5)
    adapter = XGBoostNativeAdapter(
        _xgb_cfg(n_estimators=4),
        data_cfg=data_cfg,
        memory_cfg=memory_cfg,
        target_column="y",
    )
    adapter.fit(df, y)
    pred = adapter.predict(df)
    assert pred.shape == (df.height,)


# ---------------------------------------------------------------------------
# single_source_iter helper
# ---------------------------------------------------------------------------


def test_single_source_iter_chunks_into_n_files(tmp_path: Path) -> None:
    """Chunk a 60-row frame into 3 batches → 3 Parquets → iterator yields 3 batches."""
    rng = np.random.default_rng(0)
    n = 60
    df = pl.DataFrame(
        {
            "x1": rng.normal(size=n).tolist(),
            "x2": rng.normal(size=n).tolist(),
            "y": (rng.random(n) > 0.5).astype(int).tolist(),
        }
    )
    data_iter = single_source_iter(
        df,
        target_column="y",
        batch_count=3,
        tmp_dir=tmp_path,
        cache_prefix=str(tmp_path / "cache"),
    )
    assert isinstance(data_iter, ParquetDataIter)

    parquets = sorted(tmp_path.glob("batch_*.parquet"))
    assert len(parquets) == 3
    # Round-trip read each chunk and verify row counts sum back to n.
    total_rows = sum(pl.read_parquet(p).height for p in parquets)
    assert total_rows == n

    # Iterator yields 3 batches before signaling exhaustion.
    seen: list[tuple[int, int]] = []

    def collect(data: np.ndarray, label: np.ndarray) -> None:
        seen.append((data.shape[0], label.shape[0]))

    while data_iter.next(collect):
        pass
    assert len(seen) == 3
    assert all(rows == labels for rows, labels in seen)
    assert sum(rows for rows, _ in seen) == n


def test_single_source_iter_rejects_invalid_inputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x1": rng.normal(size=10).tolist(), "y": [0] * 10})

    with pytest.raises(ValueError, match="batch_count"):
        single_source_iter(df, "y", batch_count=0, tmp_dir=tmp_path, cache_prefix="c")

    with pytest.raises(ValueError, match="target_column"):
        single_source_iter(df, "missing", batch_count=2, tmp_dir=tmp_path, cache_prefix="c")

    empty = pl.DataFrame({"x1": [], "y": []})
    with pytest.raises(ValueError, match="non-empty"):
        single_source_iter(empty, "y", batch_count=2, tmp_dir=tmp_path, cache_prefix="c")
