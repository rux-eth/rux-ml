"""PR-025 Phase 3 — empirical eval_set calibration on the crypto-h3 baseline.

Runs the same K-fold CV objective under three eval_set placements (A/B/C),
on **identical** splits + seeds + search space, then reports per-position
RMSE distributions so PR-025 Phase 4 can pick Shape 1 / 2 / 3.

Positions (per PR-022 Phase 3 + PR-025 stub):

- **A** — test fold = eval_set (current workbench default). XGBoost-internal
  ``early_stopping_rounds`` runs against the held-out fold.
- **B** — carve an inner val from the train fold via
  ``cfg.data.split_ratios.val`` (re-normalised over train+val); test fold
  is held out from XGBoost entirely. Inner-val carve respects
  ``cfg.data.split_kind`` (random / time_ordered) via :func:`make_splits`
  so PR-024's leak-prevention rule is preserved inside the CV fold.
- **C** — no ``eval_set`` passed to ``trainer.fit``; ``early_stopping_rounds``
  forced to ``None`` so XGBoost trains to ``n_estimators``. Position
  matches XGBoost's own published recommendation for CV-with-HPO.

This script is a **throwaway research artifact** committed for
reproducibility (Phase 3 receipt). The position-knob ``cv.eval_set_strategy``
is intentionally NOT defined on the workbench's config schema — Phase 4
Synthesis decides which (if any) position becomes the new default after
seeing this script's numbers.

Usage on the workbench (rux@100.90.42.41, RTX 4090)::

    cd ~/projects/rux-ml
    git checkout pr-025/cv-eval-set-design-study && git pull
    uv run python scripts/calibrate_pr025.py \
        --problem crypto_breakout_h3 \
        --study crypto_breakout_h3_baseline \
        --positions A,B,C \
        --output-json prs/PR-025-calibration-results.json

Per memory ``feedback_local_workbench_sync`` the result JSON is committed
+ pushed; local pulls it for Phase 4 analysis. The 1.4 GB parquet stays
on the workbench (not in git).
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import numpy as np
import optuna
import typer
import xgboost as xgb

from rux_ml._internal.memory import Watchdog
from rux_ml._internal.seeds import make_seed_bag
from rux_ml.config import RuxMLConfig, XGBoostTraining
from rux_ml.data import load_parquet, make_splitter, materialize
from rux_ml.training import compute_score


def _gpu_mem_used_mib() -> int:
    """Snapshot current GPU memory usage via nvidia-smi (single int call).

    Returns the highest in-use number across all visible devices; on the
    workbench's single-GPU setup that's effectively device 0.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0
    return max((int(line.strip()) for line in out.stdout.splitlines() if line.strip()), default=0)

if TYPE_CHECKING:
    from collections.abc import Generator

    import polars as pl
    from numpy.typing import NDArray

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ---------- Timing instrumentation ----------


@contextmanager
def _stopwatch(timings: dict[str, float], key: str) -> Generator[None]:
    """Accumulate per-step wallclock into ``timings[key]`` (seconds)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - t0)


# ---------- Position-specific _fold_scores variants (numpy-native) ----------
#
# Profile (2026-05-19 probe on workbench, crypto-h3 / panel_cpcv n_folds=5
# n_test_folds=2 → 10 folds per trial, 20.7M-row panel):
#   trainer_fit:   39.4s/trial (62%) — XGBoost GPU work
#   df_row_select: 14.5s/trial (23%) — pl.DataFrame[idx.tolist()] x 40
#   predict_score:  3.6s/trial  (6%) — CPU↔GPU bounce per predict
#   pipeline:       2.2s/trial  (3.5%) — sklearn Pipeline on polars
#   df_to_pandas:   0.9s/trial  (1.4%) — small, not the bottleneck
#
# Optimization (this revision):
# 1. Pre-convert x_full + y_full to numpy ONCE at the top of the objective
#    (outside the fold loop). Per-fold row select becomes a numpy slice
#    (instant) instead of polars __getitem__(list[int]).
# 2. Skip the sklearn Pipeline entirely for problems with zero
#    categorical_columns + numeric_columns subset only. Pre-select
#    numeric_columns once and call it done — this dataset has no
#    transforms beyond column selection. (General-purpose pipeline
#    handling stays in src/rux_ml/tuning/objective.py; the calibration
#    script is allowed to take the fast path for this problem because
#    eval_set placement, not pipeline cost, is what we're measuring.)
# 3. Build XGBoost QuantileDMatrix directly from numpy (no pandas
#    intermediate). Train via xgb.train (low-level) so we can construct
#    the DMatrix once per fold and reuse it for fit + predict (eliminates
#    the CPU↔GPU bounce that produced the "mismatched devices" warning).


def _xgb_cfg(cfg: RuxMLConfig) -> XGBoostTraining:
    """Narrow cfg.training to the XGBoost variant; the calibration harness
    only supports XGBoost at v0.2 (the other families don't expose the same
    early-stopping surface and aren't part of the PR-025 question)."""
    if not isinstance(cfg.training, XGBoostTraining):
        msg = (
            f"calibration harness supports only XGBoost at v0.2; "
            f"got {type(cfg.training).__name__}"
        )
        raise NotImplementedError(msg)
    return cfg.training


def _xgb_params(cfg: RuxMLConfig) -> dict[str, Any]:
    """Translate cfg.training (XGBoost variant) -> xgb.train params dict."""
    t = _xgb_cfg(cfg)
    return {
        "device": t.device,
        "tree_method": t.tree_method,
        "learning_rate": t.learning_rate,
        "max_depth": t.max_depth,
        "subsample": t.subsample,
        "colsample_bytree": t.colsample_bytree,
        "objective": "reg:squarederror" if t.metric == "rmse" else "binary:logistic",
        "eval_metric": t.metric,
        "verbosity": 0,
    }


def _rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float32]) -> float:
    diff = y_true.astype(np.float32) - y_pred
    return float(np.sqrt(np.mean(diff * diff)))


def _score(cfg: RuxMLConfig, y_true: NDArray[np.float64], y_pred: NDArray[np.float32]) -> float:
    if cfg.training.metric == "rmse":
        return _rmse(y_true, y_pred)
    # Fallback for non-rmse metrics: defer to compute_score via a tiny
    # sklearn-shim adapter. Not exercised in the crypto-h3 calibration.
    _ = compute_score  # keep import live for future-extension paths
    msg = f"calibration harness only supports rmse; got {cfg.training.metric!r}"
    raise NotImplementedError(msg)


def _carve_inner_indices(
    cfg: RuxMLConfig,
    train_idx: NDArray[np.int64],
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Position B helper: split train_idx into (inner_train_idx, inner_val_idx).

    Respects cfg.data.split_kind so the carve doesn't re-introduce the
    random-shuffle leak inside a time_ordered CV fold. Operates on
    indices only — no DataFrame materialisation.
    """
    val_ratio = cfg.data.split_ratios["val"]
    n = train_idx.size
    n_val = round(n * val_ratio)  # treat test=0 in inner carve
    if cfg.data.split_kind == "time_ordered":
        # train_idx is already sorted by the outer splitter's row order
        # (panel_cpcv yields row indices grouped by timestamp chunks); take
        # tail as val.
        return train_idx[: n - n_val], train_idx[n - n_val :]
    # random carve via deterministic shuffle.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return train_idx[perm[: n - n_val]], train_idx[perm[n - n_val :]]


def _fit_score_position_A(
    cfg: RuxMLConfig,
    x_full_np: NDArray[np.float32],
    y_full_np: NDArray[np.float64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    seed: int,
    timings: dict[str, float],
) -> float:
    with _stopwatch(timings, "df_row_select"):
        x_tr = x_full_np[train_idx]
        x_te = x_full_np[test_idx]
        y_tr = y_full_np[train_idx]
        y_te = y_full_np[test_idx]
    with _stopwatch(timings, "dmatrix_build"):
        dtrain = xgb.QuantileDMatrix(x_tr, label=y_tr)
        dtest = xgb.QuantileDMatrix(x_te, label=y_te, ref=dtrain)
    with _stopwatch(timings, "trainer_fit"):
        params = _xgb_params(cfg)
        params["seed"] = seed
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=_xgb_cfg(cfg).n_estimators,
            evals=[(dtest, "test")],
            early_stopping_rounds=_xgb_cfg(cfg).early_stopping_rounds,
            verbose_eval=False,
        )
    with _stopwatch(timings, "predict_score"):
        y_pred = booster.inplace_predict(x_te)
        score = _score(cfg, y_te, y_pred)
    return score


def _fit_score_position_B(
    cfg: RuxMLConfig,
    x_full_np: NDArray[np.float32],
    y_full_np: NDArray[np.float64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    seed: int,
    timings: dict[str, float],
) -> float:
    with _stopwatch(timings, "inner_val_carve"):
        inner_tr_idx, inner_val_idx = _carve_inner_indices(cfg, train_idx, seed)
    with _stopwatch(timings, "df_row_select"):
        x_inner_tr = x_full_np[inner_tr_idx]
        x_inner_val = x_full_np[inner_val_idx]
        x_te = x_full_np[test_idx]
        y_inner_tr = y_full_np[inner_tr_idx]
        y_inner_val = y_full_np[inner_val_idx]
        y_te = y_full_np[test_idx]
    with _stopwatch(timings, "dmatrix_build"):
        dtrain = xgb.QuantileDMatrix(x_inner_tr, label=y_inner_tr)
        dval = xgb.QuantileDMatrix(x_inner_val, label=y_inner_val, ref=dtrain)
    with _stopwatch(timings, "trainer_fit"):
        params = _xgb_params(cfg)
        params["seed"] = seed
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=_xgb_cfg(cfg).n_estimators,
            evals=[(dval, "inner_val")],
            early_stopping_rounds=_xgb_cfg(cfg).early_stopping_rounds,
            verbose_eval=False,
        )
    with _stopwatch(timings, "predict_score"):
        y_pred = booster.inplace_predict(x_te)
        score = _score(cfg, y_te, y_pred)
    return score


def _fit_score_position_C(
    cfg: RuxMLConfig,
    x_full_np: NDArray[np.float32],
    y_full_np: NDArray[np.float64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    seed: int,
    timings: dict[str, float],
) -> float:
    with _stopwatch(timings, "df_row_select"):
        x_tr = x_full_np[train_idx]
        x_te = x_full_np[test_idx]
        y_tr = y_full_np[train_idx]
        y_te = y_full_np[test_idx]
    with _stopwatch(timings, "dmatrix_build"):
        dtrain = xgb.QuantileDMatrix(x_tr, label=y_tr)
    with _stopwatch(timings, "trainer_fit"):
        params = _xgb_params(cfg)
        params["seed"] = seed
        # No eval set → no early stopping → trains to n_estimators.
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=_xgb_cfg(cfg).n_estimators,
            verbose_eval=False,
        )
    with _stopwatch(timings, "predict_score"):
        y_pred = booster.inplace_predict(x_te)
        score = _score(cfg, y_te, y_pred)
    return score


_POSITION_FN = {
    "A": _fit_score_position_A,
    "B": _fit_score_position_B,
    "C": _fit_score_position_C,
}


# ---------- Per-position study driver ----------


def _run_position(
    position: str,
    base_cfg: RuxMLConfig,
    n_trials: int,
    target_col: str,
    storage_url: str,
) -> dict[str, Any]:
    """Run an Optuna study for one position and return per-trial RMSE list."""
    df_full = materialize(load_parquet(cast("Path", base_cfg.data.source_path)))
    # PRE-CONVERT TO NUMPY ONCE. The Polars row-select bottleneck identified by
    # the 2026-05-19 probe (14.5s/trial = 23% of wallclock) lives in
    # x_full[train_idx.tolist()]; with numpy slicing on a pre-materialised
    # array, the per-fold row-select cost drops to ~0.
    #
    # Feature selection: skip the sklearn Pipeline entirely for problems with
    # zero categorical_columns. crypto-h3 fits that criterion (all 9 features
    # are numeric, no transforms). For datasets that need transforms, the
    # general-purpose pipeline path in src/rux_ml/tuning/objective.py is
    # unchanged — this script only owns the calibration harness.
    if base_cfg.features.spec.categorical_columns:
        msg = (
            "calibration harness only supports zero-categorical features at v0.2; "
            "extend the pipeline path before running on a problem with categoricals."
        )
        raise NotImplementedError(msg)
    feature_cols = list(base_cfg.features.spec.numeric_columns)
    # f32 saves VRAM + matches XGBoost-GPU's native tree-build precision.
    x_full_np: NDArray[np.float32] = (
        df_full.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    )
    y_full_np: NDArray[np.float64] = (
        df_full[target_col].to_numpy().astype(np.float64, copy=False)
    )
    # Keep a polars handle to x_full + y_full for the Splitter contract
    # (PanelCombinatorialPurgedSplitter reads time_column + asset_column
    # at split time).
    x_full: pl.DataFrame = df_full.drop(target_col)
    y_full: pl.Series = df_full[target_col]

    fit_score_fn = _POSITION_FN[position]
    study_name = f"pr025_position_{position}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
    )

    from rux_ml.tuning.objective import (  # noqa: PLC0415
        _apply_overrides,  # pyright: ignore[reportPrivateUsage]
        walk_search_space,
    )

    def objective(trial: optuna.Trial) -> float:
        overrides = walk_search_space(base_cfg.search_space, trial)
        trial_cfg = _apply_overrides(base_cfg, overrides)

        bag = make_seed_bag(
            master_entropy=trial_cfg.tuning.entropy,
            trial_number=trial.number,
        )
        timings: dict[str, float] = {}
        with _stopwatch(timings, "splitter_construct"):
            splitter = make_splitter(trial_cfg.cv, seed=bag.cv_seed)

        scores: list[float] = []
        trial_t0 = time.perf_counter()
        gpu_mib_peak = _gpu_mem_used_mib()
        with Watchdog(
            threshold_gb=trial_cfg.memory.watchdog_threshold_gb,
            sample_hz=trial_cfg.memory.watchdog_sample_hz,
        ) as wd:
            for fold_idx, (train_idx, test_idx) in enumerate(
                splitter.split(x_full, y_full)
            ):
                fold_score = fit_score_fn(
                    trial_cfg,
                    x_full_np,
                    y_full_np,
                    train_idx.astype(np.int64, copy=False),
                    test_idx.astype(np.int64, copy=False),
                    bag.xgb_seed,
                    timings,
                )
                scores.append(float(fold_score))
                # Sample GPU between folds — cheap nvidia-smi spawn.
                gpu_mib_peak = max(gpu_mib_peak, _gpu_mem_used_mib())
                trial.report(fold_score, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned
        timings["trial_total"] = time.perf_counter() - trial_t0
        # Memory peaks for the trial — host via Watchdog (per-trial sampling),
        # GPU via nvidia-smi between-fold snapshots.
        timings["peak_host_rss_mb"] = float(wd.peak_mb)
        timings["peak_gpu_mib"] = float(gpu_mib_peak)

        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("position", position)
        trial.set_user_attr("timings_s", timings)

        timing_repr = " ".join(f"{k}={v:.2f}" for k, v in sorted(timings.items()))
        typer.echo(f"    trial {trial.number} timings: {timing_repr}", err=True)
        return statistics.fmean(scores)

    t0 = time.monotonic()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.monotonic() - t0

    trial_means = [t.value for t in study.trials if t.value is not None]
    all_timings = [t.user_attrs.get("timings_s", {}) for t in study.trials]
    timing_keys: set[str] = set()
    for d in all_timings:
        timing_keys.update(d.keys())
    median_timings = {
        k: statistics.median([d.get(k, 0.0) for d in all_timings if d])
        for k in timing_keys
    }
    return {
        "position": position,
        "n_trials": n_trials,
        "trial_means": trial_means,
        "elapsed_s": elapsed,
        "best_trial_value": (
            min(trial_means) if trial_means else None
        ),  # rmse is minimize
        "all_fold_scores": [
            t.user_attrs.get("fold_scores", []) for t in study.trials
        ],
        "all_timings_s": all_timings,
        "median_timings_s": median_timings,
    }


# ---------- CLI ----------


@app.command()
def main(
    output_json: Annotated[
        Path, typer.Option("--output-json", help="Where to write the calibration JSON")
    ],
    problem: Annotated[str, typer.Option("--problem", help="Problem TOML stem")],
    study: Annotated[str, typer.Option("--study", help="Study TOML stem")],
    config: Annotated[Path, typer.Option("--config", help="Base TOML")] = Path(
        "configs/base.toml"
    ),
    positions: Annotated[
        str, typer.Option("--positions", help="Comma-separated subset of A,B,C")
    ] = "A,B,C",
    storage_url: Annotated[
        str, typer.Option("--storage-url", help="Optuna SQLite storage URL")
    ] = "sqlite:///studies/pr025_calibration.db",
    n_trials_override: Annotated[
        int,
        typer.Option(
            "--n-trials-override",
            help="If >0, use this trial count instead of cfg.tuning.n_trials (for perf probes)",
        ),
    ] = 0,
) -> None:
    base_cfg = RuxMLConfig.from_layers(config, problem=problem, study=study)
    if base_cfg.data.target_column is None:
        msg = "data.target_column must be set"
        raise typer.BadParameter(msg)
    target_col = base_cfg.data.target_column
    n_trials = n_trials_override if n_trials_override > 0 else base_cfg.tuning.n_trials

    chosen = [p.strip() for p in positions.split(",") if p.strip()]
    for p in chosen:
        if p not in _POSITION_FN:
            msg = f"unknown position {p!r}; choose from A/B/C"
            raise typer.BadParameter(msg)

    typer.echo(
        f"PR-025 calibration: problem={problem} study={study} "
        f"positions={chosen} n_trials={n_trials}"
    )

    results: dict[str, Any] = {
        "problem": problem,
        "study": study,
        "n_trials": n_trials,
        "cv_kind": base_cfg.cv.kind,
        "metric": base_cfg.training.metric,
        "positions": {},
    }
    for position in chosen:
        typer.echo(f"  ── running position {position} ──")
        res = _run_position(position, base_cfg, n_trials, target_col, storage_url)
        results["positions"][position] = res
        means = res["trial_means"]
        typer.echo(
            f"  position {position}: best={res['best_trial_value']:.5f} "
            f"median={statistics.median(means):.5f} "
            f"n_trials={len(means)} elapsed={res['elapsed_s']:.1f}s"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, default=str))
    typer.echo(f"\nwrote {output_json}")


if __name__ == "__main__":
    app()
