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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import optuna
import typer

from rux_ml._internal.env import get_versions
from rux_ml._internal.memory import Watchdog
from rux_ml._internal.seeds import make_seed_bag
from rux_ml.config import RuxMLConfig
from rux_ml.data import load_parquet, make_splits, make_splitter, materialize
from rux_ml.features import cardinalities_from, make_features
from rux_ml.training import compute_score, make_trainer

if TYPE_CHECKING:
    import polars as pl

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ---------- Position-specific _fold_scores variants ----------


def _strip_target(df: pl.DataFrame, target_col: str) -> tuple[pl.DataFrame, pl.Series]:
    return df.drop(target_col), df[target_col]


def _carve_inner_val(
    cfg: RuxMLConfig,
    x_tr: pl.DataFrame,
    y_tr: pl.Series,
    *,
    target_col: str,
    seed: int,
) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]:
    """Position B helper: split (x_tr, y_tr) into inner-train + inner-val
    using cfg.data.split_ratios.val, dispatching through make_splits so
    the carve respects cfg.data.split_kind (random vs time_ordered)."""
    val_ratio = cfg.data.split_ratios["val"]
    # Re-normalise over train+val only; test=0 here (test fold is the
    # outer CV's test_idx, not carved from the inner train fold).
    inner_ratios = {
        "train": 1.0 - val_ratio,
        "val": val_ratio,
        "test": 0.0,
    }
    # Glue target back temporarily so make_splits sees a single DataFrame.
    merged = x_tr.with_columns(y_tr.alias(target_col))
    # Temporarily override the split_ratios for the inner carve only.
    inner_cfg = cfg.model_copy(deep=True)
    inner_cfg.data.split_ratios = inner_ratios
    parts = make_splits(inner_cfg, merged, seed=seed)
    x_inner_tr, y_inner_tr = _strip_target(parts["train"], target_col)
    x_inner_val, y_inner_val = _strip_target(parts["val"], target_col)
    return x_inner_tr, y_inner_tr, x_inner_val, y_inner_val


def _fit_score_position_A(
    cfg: RuxMLConfig,
    x_tr: pl.DataFrame,
    y_tr: pl.Series,
    x_te: pl.DataFrame,
    y_te: pl.Series,
    seed: int,
    target_col: str,
) -> float:
    _ = (seed, target_col)
    cards = cardinalities_from(x_tr, cfg.features.spec.categorical_columns)
    pipeline = make_features(cfg.features, cardinalities=cards if cards else None)
    pipeline.fit(x_tr, y_tr.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    x_tr_t = cast("pl.DataFrame", pipeline.transform(x_tr))  # pyright: ignore[reportUnknownMemberType]
    x_te_t = cast("pl.DataFrame", pipeline.transform(x_te))  # pyright: ignore[reportUnknownMemberType]

    trainer = make_trainer(cfg.training, seed=seed)
    trainer.fit(
        x_tr_t.to_pandas(),
        y_tr.to_numpy(),
        eval_set=[(x_te_t.to_pandas(), y_te.to_numpy())],
        verbose=False,
    )
    return compute_score(cfg.training.metric, trainer, x_te_t.to_pandas(), y_te.to_numpy())


def _fit_score_position_B(
    cfg: RuxMLConfig,
    x_tr: pl.DataFrame,
    y_tr: pl.Series,
    x_te: pl.DataFrame,
    y_te: pl.Series,
    seed: int,
    target_col: str,
) -> float:
    # Inner-val carve from train fold; test fold is held out from XGBoost.
    x_inner_tr, y_inner_tr, x_inner_val, y_inner_val = _carve_inner_val(
        cfg, x_tr, y_tr, target_col=target_col, seed=seed
    )
    cards = cardinalities_from(x_inner_tr, cfg.features.spec.categorical_columns)
    pipeline = make_features(cfg.features, cardinalities=cards if cards else None)
    pipeline.fit(x_inner_tr, y_inner_tr.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    x_inner_tr_t = cast("pl.DataFrame", pipeline.transform(x_inner_tr))  # pyright: ignore[reportUnknownMemberType]
    x_inner_val_t = cast("pl.DataFrame", pipeline.transform(x_inner_val))  # pyright: ignore[reportUnknownMemberType]
    x_te_t = cast("pl.DataFrame", pipeline.transform(x_te))  # pyright: ignore[reportUnknownMemberType]

    trainer = make_trainer(cfg.training, seed=seed)
    trainer.fit(
        x_inner_tr_t.to_pandas(),
        y_inner_tr.to_numpy(),
        eval_set=[(x_inner_val_t.to_pandas(), y_inner_val.to_numpy())],
        verbose=False,
    )
    return compute_score(cfg.training.metric, trainer, x_te_t.to_pandas(), y_te.to_numpy())


def _fit_score_position_C(
    cfg: RuxMLConfig,
    x_tr: pl.DataFrame,
    y_tr: pl.Series,
    x_te: pl.DataFrame,
    y_te: pl.Series,
    seed: int,
    target_col: str,
) -> float:
    _ = target_col
    # Override early_stopping_rounds=None for this trainer instance only.
    cfg_C = cfg.model_copy(deep=True)
    cfg_C.training.early_stopping_rounds = None

    cards = cardinalities_from(x_tr, cfg_C.features.spec.categorical_columns)
    pipeline = make_features(cfg_C.features, cardinalities=cards if cards else None)
    pipeline.fit(x_tr, y_tr.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    x_tr_t = cast("pl.DataFrame", pipeline.transform(x_tr))  # pyright: ignore[reportUnknownMemberType]
    x_te_t = cast("pl.DataFrame", pipeline.transform(x_te))  # pyright: ignore[reportUnknownMemberType]

    trainer = make_trainer(cfg_C.training, seed=seed)
    trainer.fit(
        x_tr_t.to_pandas(),
        y_tr.to_numpy(),
        # No eval_set → no early stopping → trains to n_estimators.
        verbose=False,
    )
    return compute_score(cfg_C.training.metric, trainer, x_te_t.to_pandas(), y_te.to_numpy())


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
    x_full, y_full = _strip_target(df_full, target_col)
    versions = get_versions(base_cfg.memory)
    _ = versions  # captured for completeness; not asserted in this throwaway script

    fit_score_fn = _POSITION_FN[position]
    # Use distinct study names so positions don't share trials.
    study_name = f"pr025_position_{position}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
    )

    # Walk the search space ONCE so all positions explore the same grid;
    # mirror PR-007's objective shape but with the position-specific
    # inner fit-and-score function.
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
        splitter = make_splitter(trial_cfg.cv, seed=bag.cv_seed)

        scores: list[float] = []
        with Watchdog(
            threshold_gb=trial_cfg.memory.watchdog_threshold_gb,
            sample_hz=trial_cfg.memory.watchdog_sample_hz,
        ):
            for fold_idx, (train_idx, test_idx) in enumerate(
                splitter.split(x_full, y_full)
            ):
                x_tr = x_full[train_idx.tolist()]
                x_te = x_full[test_idx.tolist()]
                y_tr = y_full[train_idx.tolist()]
                y_te = y_full[test_idx.tolist()]

                fold_score = fit_score_fn(
                    trial_cfg, x_tr, y_tr, x_te, y_te, bag.xgb_seed, target_col
                )
                scores.append(float(fold_score))
                trial.report(fold_score, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned

        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("position", position)
        return statistics.fmean(scores)

    t0 = time.monotonic()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.monotonic() - t0

    trial_means = [t.value for t in study.trials if t.value is not None]
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
) -> None:
    base_cfg = RuxMLConfig.from_layers(config, problem=problem, study=study)
    if base_cfg.data.target_column is None:
        msg = "data.target_column must be set"
        raise typer.BadParameter(msg)
    target_col = base_cfg.data.target_column
    n_trials = base_cfg.tuning.n_trials

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
