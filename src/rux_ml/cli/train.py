"""``rux-ml train`` — single baseline training end-to-end (per PR-006 + D7 + PR-013).

Wraps a one-off training as a 1-trial Optuna study so the same SQLite store
holds both sweeps and baselines (per D7). The provenance triple now includes
the full PR-013 environment block (``entropy_hex``, ``image_digest``, library
+ CUDA versions, ``omp_threads``) alongside the PR-006 minimum (config hashes
+ ``data_hash`` + ``git_sha``) and the PR-011 ``peak_rss_mb``.

PR-009 refactored this to use the shared ``runs.ask_tell.one_off_run``
context manager and ``runs.attrs.TrialAttrs.from_cfg(...).record(trial)`` so
sweep and one-off paths funnel through the same provenance recorder.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import optuna
import polars as pl
import typer

from rux_ml._internal.env import EnvironmentVersions, get_versions, pin_threads
from rux_ml._internal.memory import MemoryPressureError, Watchdog
from rux_ml._internal.seeds import SeedBag, make_seed_bag
from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig, XGBoostTraining
from rux_ml.data import load_parquet, make_splits, materialize
from rux_ml.features import cardinalities_from, make_features
from rux_ml.runs import (
    TrialAttrs,
    data_hashes,
    one_off_run,
)
from rux_ml.training import (
    XGBoostNativeAdapter,
    compute_score,
    estimate_x_bytes,
    make_trainer,
    optuna_direction,
    select_ingest,
)


def _require(cfg: RuxMLConfig) -> tuple[Path, str]:
    if cfg.data.source_path is None or cfg.data.target_column is None:
        msg = (
            "rux-ml train requires data.source_path and data.target_column to be set "
            "(via configs/problems/<problem>.toml or --set data.source_path=… "
            "--set data.target_column=…)"
        )
        raise typer.BadParameter(msg)
    return cfg.data.source_path, cfg.data.target_column


def _strip_target(df: pl.DataFrame, target_col: str) -> tuple[pl.DataFrame, pl.Series]:
    return df.drop(target_col), df[target_col]


def _fit_and_score(
    cfg: RuxMLConfig, source_path: Path, target_col: str, bag: SeedBag
) -> tuple[float, int | None]:
    """Fit + score one baseline using PR-013-derived seeds for split + trainer.

    The one-off baseline shares the data-fold layout with the registry-side
    re-fit at promote time (both consume :func:`make_splits` with the same
    ``bag.split_seed``), so a promoted bundle reproduces the exact baseline
    configuration the user saw at training. ``make_splits`` dispatches
    between random and temporal split per ``cfg.data.split_kind`` (PR-024);
    the temporal path is deterministic and ignores the seed, so the same
    reproducibility contract holds.
    """
    df = materialize(load_parquet(source_path))
    splits = make_splits(cfg, df, seed=bag.split_seed)
    x_train, y_train = _strip_target(splits["train"], target_col)
    x_val, y_val = _strip_target(splits["val"], target_col)

    cards = cardinalities_from(x_train, cfg.features.spec.categorical_columns)
    pipeline = make_features(cfg.features, cardinalities=cards if cards else None)
    pipeline.fit(x_train, y_train.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    # sklearn Pipeline.transform stubs return ``Unknown``; PR-005's terminal
    # ``to_polars`` step guarantees a Polars DataFrame at the boundary.
    x_train_t = cast("pl.DataFrame", pipeline.transform(x_train))  # pyright: ignore[reportUnknownMemberType]
    x_val_t = cast("pl.DataFrame", pipeline.transform(x_val))  # pyright: ignore[reportUnknownMemberType]

    # Decision-rule classification: PR-006 logs the chosen DMatrix class so
    # operators see which path is in use. PR-033 wires the return value
    # through into actual DMatrix construction when ``cfg.training.use_native``
    # is True (opt-in only; the sklearn-wrapper path remains the default and
    # is the only one exercised in HPO per ``_check_extmem_compat``).
    dmatrix_cls = select_ingest(estimate_x_bytes(x_train_t), cfg.data)
    use_native = isinstance(cfg.training, XGBoostTraining) and cfg.training.use_native
    path_label = "native API" if use_native else "sklearn wrapper"
    typer.echo(f"  ingest path: {dmatrix_cls.__name__} ({path_label})", err=True)

    if use_native:
        # Cast is safe under the isinstance check above; mypy/basedpyright
        # narrow ``cfg.training`` to ``XGBoostTraining`` here.
        xgb_cfg = cast("XGBoostTraining", cfg.training)
        adapter = XGBoostNativeAdapter(
            xgb_cfg,
            data_cfg=cfg.data,
            memory_cfg=cfg.memory,
            target_column=target_col,
            seed=bag.xgb_seed,
        )
        adapter.fit(
            x_train_t,
            y_train,
            eval_set=[(x_val_t, y_val)],
            verbose=False,
        )
        score = compute_score(
            cfg.training.metric, adapter, x_val_t.to_pandas(), y_val.to_numpy()
        )
        best_iter = adapter.best_iteration
        return score, int(best_iter) if best_iter is not None else None

    trainer = make_trainer(cfg.training, seed=bag.xgb_seed)
    x_train_pd = x_train_t.to_pandas()
    x_val_pd = x_val_t.to_pandas()
    trainer.fit(
        x_train_pd,
        y_train.to_numpy(),
        eval_set=[(x_val_pd, y_val.to_numpy())],
        verbose=False,
    )
    score = compute_score(cfg.training.metric, trainer, x_val_pd, y_val.to_numpy())
    best_iter = getattr(trainer, "best_iteration", None)
    return score, int(best_iter) if best_iter is not None else None


def _record_attrs(
    cfg: RuxMLConfig,
    hashes: dict[str, str],
    *,
    bag: SeedBag,
    versions: EnvironmentVersions,
    peak_rss_mb: float,
    best_iteration: int | None,
    trial: optuna.Trial,
) -> None:
    TrialAttrs.from_cfg(
        cfg,
        hashes,
        metric=cfg.training.metric,
        peak_rss_mb=peak_rss_mb,
        bag=bag,
        versions=versions,
        best_iteration=best_iteration,
    ).record(trial)


def run_command(ctx: typer.Context) -> None:
    """Run a single baseline training; recorded as a 1-trial Optuna study (per D7)."""
    opts = get_options(ctx)
    cfg = RuxMLConfig.from_layers(
        opts.config,
        problem=opts.problem,
        study=opts.study,
        overrides=opts.overrides,
    )
    # PR-011: pin OMP/BLAS/POLARS env vars before any in-process fit (no-op for the
    # subprocess child case which self-pins via ``_internal/trial_runner.main``).
    pin_threads(cfg.memory)

    source_path, target_col = _require(cfg)
    hashes = data_hashes(source_path)
    versions = get_versions(cfg.memory)

    with one_off_run(
        cfg,
        problem=opts.problem,
        study=opts.study,
        direction=optuna_direction(cfg.training.metric),
    ) as run:
        # PR-013: derive per-trial bag using the one-off trial's number
        # (typically 0 in a fresh study; non-zero when --study targets an
        # existing study and one_off_run appends a new trial). The bag is
        # stable for (master_entropy, trial.number) so reruns are reproducible.
        bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=run.trial.number)

        # PR-011: watchdog wraps the fit; tripped → pruned.
        with Watchdog(
            threshold_gb=cfg.memory.watchdog_threshold_gb,
            sample_hz=cfg.memory.watchdog_sample_hz,
        ) as wd:
            try:
                score, best_iter = _fit_and_score(cfg, source_path, target_col, bag)
            except MemoryPressureError:
                _record_attrs(
                    cfg,
                    hashes,
                    bag=bag,
                    versions=versions,
                    peak_rss_mb=wd.peak_mb,
                    best_iteration=None,
                    trial=run.trial,
                )
                raise optuna.TrialPruned from None

        _record_attrs(
            cfg,
            hashes,
            bag=bag,
            versions=versions,
            peak_rss_mb=wd.peak_mb,
            best_iteration=best_iter,
            trial=run.trial,
        )
        if wd.tripped:
            # Threshold crossed during the fit even though the fit completed —
            # mark the trial pruned so it doesn't pollute the best-trial pool.
            raise optuna.TrialPruned
        run.tell(score)

        typer.echo(f"score ({cfg.training.metric}): {score:.6f}")
        typer.echo(f"  study:    {run.study.study_name}")
        typer.echo(f"  trial:    {run.trial.number}")
        typer.echo(f"  storage:  {cfg.runs.storage_url}")
        typer.echo(f"  peak_rss_mb: {wd.peak_mb:.1f}")
        typer.echo(f"  entropy_hex: {bag.entropy_hex}")
        if best_iter is not None:
            typer.echo(f"  best_iter: {best_iter}")
