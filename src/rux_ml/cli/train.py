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

import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import polars as pl
import typer

from rux_ml._internal.env import EnvironmentVersions, get_versions, pin_threads
from rux_ml._internal.memory import MemoryPressureError, Watchdog
from rux_ml._internal.seeds import SeedBag, make_seed_bag
from rux_ml.cli._shared import get_options, refuse_oracle_source
from rux_ml.config import RuxMLConfig, XGBoostTraining
from rux_ml.config.m9_gates import M9Gates, M9GatesError, load_m9_gates
from rux_ml.data import load_parquet, make_splits, materialize, split_definition
from rux_ml.data.leakage import LeakageError, assert_regime_clean, leakage_audit
from rux_ml.features import cardinalities_from, make_features
from rux_ml.runs import (
    TrialAttrs,
    build_metrics_dict,
    data_hashes,
    make_artifact_store,
    one_off_run,
    upload_diagnostics,
)
from rux_ml.training import (
    XGBoostNativeAdapter,
    compute_score,
    estimate_x_bytes,
    make_trainer,
    optuna_direction,
    select_ingest,
)
from rux_ml.training.honesty import signed_error_honesty


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


def _label_diagnostics(
    splits: dict[str, pl.DataFrame], columns: list[str]
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Summarise ``[m9] diagnostic_columns`` per one-off fold (PR-041; D45 #3).

    The other labels of an M9 table ride beside the target as diagnostics —
    ``{fold: {column: {n, null_count, mean, std, min, max}}}`` in the ``fold_meta``
    sidecar — and are never trained on. ``n`` counts values; ``null_count`` counts
    nulls and NaNs (a label undefined on a row, e.g. a markout on an unfilled rung);
    a statistic with no values behind it is ``None``, never a fabricated 0.0. A
    listed column absent from the set refuses the run.
    """
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for fold in ("train", "val"):
        df = splits[fold]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            msg = f"m9.diagnostic_columns {missing} not in the training set columns {df.columns}"
            raise typer.BadParameter(msg)
        out[fold] = {}
        for c in columns:
            raw = df[c].cast(pl.Float64)
            s = raw.fill_nan(None).drop_nulls()

            def _stat(v: object) -> float | None:
                return None if v is None else float(v)  # pyright: ignore[reportArgumentType]

            out[fold][c] = {
                "n": float(s.len()),
                "null_count": float(raw.len() - s.len()),
                "mean": _stat(s.mean()),
                "std": _stat(s.std()),
                "min": _stat(s.min()),
                "max": _stat(s.max()),
            }
    return out


def _leakage(cfg: RuxMLConfig, splits: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """PR-045: audit the split (program C9's leakage tests) and, for an ``[m9]`` fit,
    refuse a regime whose separation the audit does not find (exit 2).

    The stamp window is the label's reach (``[m9] h_max_ms``, program PR-024 A3),
    else the split embargo; every fit records the audit, only ``[m9]`` fits enforce it (a
    legacy ``time_ordered`` problem with no embargo promises no stamp separation).
    """
    window = cfg.m9.h_max_ms if cfg.m9 is not None else None
    if window is None:
        window = cfg.data.split_embargo
    audit = leakage_audit(
        splits,
        time_column=cfg.data.time_column,
        group_column=cfg.data.group_column,
        window=window,
    )
    if cfg.m9 is not None:
        try:
            assert_regime_clean(cfg.data.split_kind, audit)
        except LeakageError as exc:
            raise typer.BadParameter(str(exc)) from exc
    return audit


def _m9_gates(cfg: RuxMLConfig) -> M9Gates | None:
    """PR-044: read the operator-signed M9 keys before any trial row (exit 2 on doubt)."""
    if cfg.m9 is None or cfg.m9.gates is None:
        return None
    try:
        return load_m9_gates(cfg.m9.gates)
    except M9GatesError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _oos_record(
    cfg: RuxMLConfig,
    model: Any,
    pipeline: Any,
    test: pl.DataFrame,
    target_col: str,
    gates: M9Gates | None,
) -> dict[str, Any]:
    """PR-044: an ``[m9]`` fit scored on the held-out ``test`` partition.

    The workbench metric (Brier for the fill fraction, MAE in bp for markout and
    walk — program D45 #3) on the rows with a realized label, and, with
    ``[m9] signed_error_honesty``, the honesty test at the signed thresholds
    (:func:`rux_ml.training.honesty.signed_error_honesty`) with the gates file's
    sha256. The partition was not used to fit or to early-stop.
    """
    out: dict[str, Any] = {"partition": "test", "n_rows": test.height}
    if gates is not None:
        out["gates_sha256"] = gates.sha256
        out["gates_path"] = gates.path
    if test.height == 0:
        return out
    x_test, y_test = _strip_target(test, target_col)
    x_pd = cast("pl.DataFrame", pipeline.transform(x_test)).to_pandas()  # pyright: ignore[reportUnknownMemberType]
    realized = y_test.cast(pl.Float64).fill_nan(None).to_numpy()
    keep = ~np.isnan(realized)
    out["n_scored"] = int(keep.sum())
    if keep.any():
        out["metric"] = cfg.training.metric
        out["score"] = compute_score(cfg.training.metric, model, x_pd[keep], realized[keep])
    if cfg.m9 is not None and cfg.m9.signed_error_honesty:
        assert gates is not None  # M9Config validator: honesty requires [m9.gates]
        out["honesty"] = signed_error_honesty(
            model.predict(x_pd),
            realized,
            no_underdeduct_frac_min=gates.no_underdeduct_frac_min,
            overdeduct_max_rel=gates.overdeduct_max_rel,
        )
    return out


def _fit_and_score(
    cfg: RuxMLConfig,
    source_path: Path,
    target_col: str,
    bag: SeedBag,
    gates: M9Gates | None = None,
) -> tuple[float, int | None, float, int, dict[str, Any]]:
    """Fit + score one baseline using PR-013-derived seeds for split + trainer.

    Returns ``(score, best_iter, fit_seconds, train_row_count, fold_extras)`` —
    the middle two added by PR-034 so the diagnostic-artifact upload in
    ``run_command`` can populate ``fold_meta.json``'s single-fold entry without
    re-doing the measurement; ``fold_extras`` holds ``diagnostics`` (PR-041, the
    per-fold summary of ``[m9] diagnostic_columns``, empty without the layer)
    and ``split_definition`` (PR-042, :func:`rux_ml.data.split_definition`).

    The one-off baseline shares the data-fold layout with the registry-side
    re-fit at promote time (both consume :func:`make_splits` with the same
    ``bag.split_seed``), so a promoted bundle reproduces the exact baseline
    configuration the user saw at training. ``make_splits`` dispatches
    between random and temporal split per ``cfg.data.split_kind`` (PR-024);
    the temporal path is deterministic and ignores the seed, so the same
    reproducibility contract holds.
    """
    df = materialize(load_parquet(source_path, oracle=cfg.data.oracle))
    splits = make_splits(cfg, df, seed=bag.split_seed)
    fold_extras: dict[str, Any] = {
        "diagnostics": (
            _label_diagnostics(splits, cfg.m9.diagnostic_columns) if cfg.m9 is not None else {}
        ),
        "split_definition": split_definition(cfg, df, splits),
        "leakage": _leakage(cfg, splits),
    }
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

    train_row_count = int(x_train_t.height)
    fit_start = time.perf_counter()

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
        score = compute_score(cfg.training.metric, adapter, x_val_t.to_pandas(), y_val.to_numpy())
        fit_seconds = time.perf_counter() - fit_start
        best_iter = adapter.best_iteration
        if cfg.m9 is not None:
            fold_extras["oos"] = _oos_record(
                cfg, adapter, pipeline, splits["test"], target_col, gates
            )
        return (
            score,
            int(best_iter) if best_iter is not None else None,
            float(fit_seconds),
            train_row_count,
            fold_extras,
        )

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
    fit_seconds = time.perf_counter() - fit_start
    best_iter = getattr(trainer, "best_iteration", None)
    if cfg.m9 is not None:
        fold_extras["oos"] = _oos_record(cfg, trainer, pipeline, splits["test"], target_col, gates)
    return (
        score,
        int(best_iter) if best_iter is not None else None,
        float(fit_seconds),
        train_row_count,
        fold_extras,
    )


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
    refuse_oracle_source(cfg)  # PR-040: before hashing or creating the trial row
    gates = _m9_gates(cfg)  # PR-044: the signed M9 keys, before the trial row too
    hashes = data_hashes(source_path, oracle=cfg.data.oracle)
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
                score, best_iter, fit_seconds, train_row_count, fold_extras = _fit_and_score(
                    cfg, source_path, target_col, bag, gates
                )
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

        # PR-034: per-trial diagnostic artifact upload — in-objective, post-fit,
        # no try/except guard (mirrors tuning/objective.py site). Single-fold
        # semantics (n_folds=1) for the one-off baseline.
        artifact_store = make_artifact_store(cfg, study_name=run.study.study_name)
        metrics_dict = build_metrics_dict(cfg.training.metric, [score], peak_rss_mb=wd.peak_mb)
        fold_meta: list[dict[str, Any]] = [
            {
                "fold_idx": 0,
                "row_count": train_row_count,
                "fit_seconds": fit_seconds,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                # PR-041: the other M9 labels, summarised per fold (empty without [m9]);
                # PR-042: the split definition (holdout groups / embargo + purged rows).
                **fold_extras,
            }
        ]
        with tempfile.TemporaryDirectory(prefix="rux_ml_artifacts_") as tmp:
            upload_diagnostics(
                run.trial,
                artifact_store,
                metrics=metrics_dict,
                fold_meta=fold_meta,
                tmp_dir=Path(tmp),
            )
        run.tell(score)

        typer.echo(f"score ({cfg.training.metric}): {score:.6f}")
        typer.echo(f"  study:    {run.study.study_name}")
        typer.echo(f"  trial:    {run.trial.number}")
        typer.echo(f"  storage:  {cfg.runs.storage_url}")
        typer.echo(f"  peak_rss_mb: {wd.peak_mb:.1f}")
        typer.echo(f"  entropy_hex: {bag.entropy_hex}")
        if best_iter is not None:
            typer.echo(f"  best_iter: {best_iter}")
