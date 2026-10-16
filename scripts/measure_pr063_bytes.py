# pyright: reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""PR-063 Q4: the native adapter's model bytes, matrices at XGBoost's default
``nthread`` (``dev``) vs ``nthread = memory.omp_threads`` (the branch).

Run once per code version in the same environment, the four thread vars UNSET
(so XGBoost's matrix default is the OpenMP default: 32 on the desktop); no pin
is called. Compare the printed digests across the two runs. Each fit runs twice
per run (self-reproduction). Prints one JSON object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from threadpoolctl import threadpool_info

from rux_ml.config import DataConfig
from rux_ml.config.memory import MemoryConfig
from rux_ml.training import XGBoostNativeAdapter, XGBoostTraining
from rux_ml.training.xgboost import native_adapter


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _data(n: int, f: int, seed: int, *, binary: bool) -> tuple[pl.DataFrame, Any]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f)).astype(np.float32)
    y = x @ rng.normal(size=f) + rng.normal(0, 0.5, size=n)
    if binary:
        y = (y > 0).astype(int)
    return pl.DataFrame({f"feat__{i}": x[:, i] for i in range(f)}), y


def _fit(device: str, path: str, *, binary: bool, n: int, f: int, rounds: int) -> dict[str, Any]:
    x, y = _data(n, f, 0, binary=binary)
    xv, yv = _data(n // 4, f, 1, binary=binary)
    cfg = XGBoostTraining(
        device=device,  # pyright: ignore[reportArgumentType]
        metric="logloss" if binary else "rmse",  # pyright: ignore[reportArgumentType]
        n_estimators=rounds,
        max_depth=6,
        learning_rate=0.1,
        early_stopping_rounds=None,
    )
    adapter = XGBoostNativeAdapter(
        cfg, data_cfg=DataConfig(), memory_cfg=MemoryConfig(), target_column="y", seed=7
    )
    original = native_adapter.select_ingest
    if path == "extmem":
        native_adapter.select_ingest = lambda *_a, **_k: xgb.ExtMemQuantileDMatrix  # pyright: ignore[reportAttributeAccessIssue]
    try:
        adapter.fit(x, y, eval_set=[(xv, yv)])
    finally:
        native_adapter.select_ingest = original
    assert adapter.booster is not None
    pred = np.ascontiguousarray(adapter.predict(xv.to_pandas()))
    return {
        "model_sha256": _sha(bytes(adapter.booster.save_raw("ubj"))),
        "pred_sha256": _sha(pred.tobytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--features", type=int, default=27)
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()
    use_cuda = bool(xgb.build_info().get("USE_CUDA"))  # pyright: ignore[reportUnknownMemberType]
    arms = [("cpu", "qdm"), ("cpu", "extmem")] + ([("cuda", "qdm")] if use_cuda else [])
    out: dict[str, Any] = {
        "label": args.label,
        "matrix_nthread_explicit": hasattr(XGBoostNativeAdapter, "_matrix_nthread"),
        "memory_omp_threads": MemoryConfig().omp_threads,
        "xgboost": xgb.__version__,
        "fits": {},
    }
    for device, path in arms:
        for binary in (False, True):
            key = f"{device}/{path}/{'binary' if binary else 'regression'}"
            try:
                reps = [
                    _fit(
                        device,
                        path,
                        binary=binary,
                        n=args.rows,
                        f=args.features,
                        rounds=args.rounds,
                    )
                    for _ in range(2)
                ]
                out["fits"][key] = {**reps[0], "self_repro": reps[0] == reps[1]}
            except Exception as exc:  # recorded per fit, not hidden
                out["fits"][key] = {"error": repr(exc)}
    out["openmp_after"] = sorted(
        {i["num_threads"] for i in threadpool_info() if i["user_api"] == "openmp"}
    )
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
