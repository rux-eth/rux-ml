"""``rux-ml tune`` verb group — Optuna sweeps.

Sequential trials, SQLite-backed. ``cfg.tuning.trial_isolation`` (PR-008)
selects the dispatch path:

- ``"subprocess"`` (default per D10/D16) — parent loops ``n_trials`` times
  invoking ``python -m rux_ml._internal.trial_runner`` via ``subprocess.run``
  with spawn semantics (CUDA + fork is forbidden per ``docs/CONSTRAINTS.md``).
  Storage coordinates state across children.
- ``"in_process"`` — parent runs ``study.optimize(objective, n_trials=N)``
  directly. Faster startup; Python may not reclaim RAM between trials
  (Optuna #1178). Useful for debugging.

Per PR-007 Tier-2 research, the objective is **K-fold CV-mean** consuming
PR-015's Splitter; default sampler is ``TPESampler``; default pruner is
``WilcoxonPruner``.
"""

from __future__ import annotations

from typing import Annotated

import optuna
import typer

from rux_ml._internal.env import pin_threads
from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig
from rux_ml.runs import study_name
from rux_ml.training import optuna_direction
from rux_ml.tuning import (
    build_objective,
    create_or_load,
    make_pruner,
    make_sampler,
    run_subprocess_trial,
)

app = typer.Typer(
    name="tune",
    no_args_is_help=True,
    help="Optuna hyperparameter sweeps.",
)


def _load_cfg(
    ctx: typer.Context,
) -> tuple[RuxMLConfig, str | None, str | None, dict[str, object]]:
    opts = get_options(ctx)
    cfg = RuxMLConfig.from_layers(
        opts.config,
        problem=opts.problem,
        study=opts.study,
        overrides=opts.overrides,
    )
    return cfg, opts.problem, opts.study, dict(opts.overrides)


def _resolve_study_name(
    explicit: str | None,
    cfg: RuxMLConfig,
    problem: str | None,
    study: str | None,
) -> str:
    """Use the explicit CLI argument when provided; otherwise fall back to the templated name."""
    return explicit or study_name(cfg, problem=problem, study=study)


def _open_study(cfg: RuxMLConfig, name: str) -> optuna.Study:
    return create_or_load(
        name=name,
        storage=cfg.runs.storage_url,
        sampler=make_sampler(cfg.tuning, seed=cfg.tuning.entropy),
        pruner=make_pruner(cfg.tuning),
        direction=optuna_direction(cfg.training.metric),
        load_if_exists=True,
    )


def _run_trials(
    ctx: typer.Context,
    cfg: RuxMLConfig,
    problem: str | None,
    study_layer: str | None,
    overrides: dict[str, object],
    study_obj: optuna.Study,
    name: str,
    n_trials: int,
) -> None:
    """Dispatch trials per ``cfg.tuning.trial_isolation``."""
    opts = get_options(ctx)
    if cfg.tuning.trial_isolation == "in_process":
        # PR-011: parent pins env vars before the in-process fit loop (subprocess
        # children already self-pin via ``_internal/trial_runner.main``).
        pin_threads(cfg.memory)
        objective = build_objective(cfg)
        study_obj.optimize(objective, n_trials=n_trials)
        return

    # Subprocess path: parent loop, each iteration spawns a fresh child.
    for i in range(n_trials):
        typer.echo(f"  [trial {i + 1}/{n_trials}] dispatching subprocess child...", err=True)
        rc = run_subprocess_trial(
            cfg=cfg,
            cfg_path=opts.config,
            problem=problem,
            study_layer=study_layer,
            study_name=name,
            overrides=overrides,
        )
        if rc != 0:
            typer.echo(
                f"  [trial {i + 1}/{n_trials}] child exited with code {rc}; "
                f"continuing to next trial",
                err=True,
            )


@app.command(name="start")
def start(
    ctx: typer.Context,
    study_name_arg: Annotated[
        str | None,
        typer.Argument(
            metavar="STUDY_NAME",
            help="Optional study identifier; defaults to runs.study_name_template substitution.",
        ),
    ] = None,
    n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50,
) -> None:
    """Create or load a study and run N trials."""
    cfg, problem, study_layer, overrides = _load_cfg(ctx)
    name = _resolve_study_name(study_name_arg, cfg, problem, study_layer)
    study_obj = _open_study(cfg, name)
    typer.echo(f"study:    {name}")
    typer.echo(f"storage:  {cfg.runs.storage_url}")
    typer.echo(f"sampler:  {cfg.tuning.sampler}    pruner: {cfg.tuning.pruner}")
    typer.echo(f"isolation: {cfg.tuning.trial_isolation}    n_trials: {n_trials}")
    _run_trials(ctx, cfg, problem, study_layer, overrides, study_obj, name, n_trials)
    # Reload to see the children's writes (subprocess path); harmless on in-process.
    study_obj = optuna.load_study(study_name=name, storage=cfg.runs.storage_url)
    completed = [t for t in study_obj.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        typer.echo(f"\nbest value ({cfg.training.metric}): {study_obj.best_value:.6f}")
        typer.echo(f"best trial:  #{study_obj.best_trial.number}")
        typer.echo(f"best params: {study_obj.best_trial.params}")


@app.command(name="resume")
def resume(
    ctx: typer.Context,
    study_name_arg: Annotated[str, typer.Argument(metavar="STUDY_NAME")],
    n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50,
) -> None:
    """Add N more trials to an existing study (idempotent ``load_if_exists=True``)."""
    cfg, problem, study_layer, overrides = _load_cfg(ctx)
    study_obj = _open_study(cfg, study_name_arg)
    typer.echo(f"resuming study {study_name_arg} ({len(study_obj.trials)} existing trials)")
    typer.echo(f"isolation: {cfg.tuning.trial_isolation}    n_trials: {n_trials}")
    _run_trials(
        ctx, cfg, problem, study_layer, overrides, study_obj, study_name_arg, n_trials
    )
    study_obj = optuna.load_study(study_name=study_name_arg, storage=cfg.runs.storage_url)
    completed = [t for t in study_obj.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_value = study_obj.best_value if completed else float("nan")
    typer.echo(
        f"\ntotal trials: {len(study_obj.trials)} "
        f"(best {cfg.training.metric}={best_value:.6f})"
    )


@app.command(name="status")
def status(
    ctx: typer.Context,
    study_name_arg: Annotated[str, typer.Argument(metavar="STUDY_NAME")],
) -> None:
    """Print progress, current best, and best trial's user_attrs."""
    cfg, _, _, _ = _load_cfg(ctx)
    try:
        study_obj = optuna.load_study(study_name=study_name_arg, storage=cfg.runs.storage_url)
    except KeyError as exc:
        msg = f"study {study_name_arg!r} not found at {cfg.runs.storage_url}"
        raise typer.BadParameter(msg) from exc

    completed = [t for t in study_obj.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study_obj.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study_obj.trials if t.state == optuna.trial.TrialState.FAIL]

    typer.echo(f"study:    {study_name_arg}")
    typer.echo(f"storage:  {cfg.runs.storage_url}")
    typer.echo(f"trials:   total={len(study_obj.trials)} ")
    typer.echo(f"          completed={len(completed)} pruned={len(pruned)} failed={len(failed)}")
    if completed:
        typer.echo(f"\nbest value ({cfg.training.metric}): {study_obj.best_value:.6f}")
        typer.echo(f"best trial:  #{study_obj.best_trial.number}")
        typer.echo(f"best params: {study_obj.best_trial.params}")
        typer.echo("best user_attrs:")
        for key, value in sorted(study_obj.best_trial.user_attrs.items()):
            typer.echo(f"  {key}: {value}")


@app.command(name="retry-trial")
def retry_trial(
    ctx: typer.Context,
    study_name_arg: Annotated[str, typer.Argument(metavar="STUDY_NAME")],
    trial_id: Annotated[int, typer.Argument()],
) -> None:
    """Re-enqueue a failed trial with its original params via ``study.add_trial``."""
    cfg, _, _, _ = _load_cfg(ctx)
    try:
        study_obj = optuna.load_study(study_name=study_name_arg, storage=cfg.runs.storage_url)
    except KeyError as exc:
        msg = f"study {study_name_arg!r} not found at {cfg.runs.storage_url}"
        raise typer.BadParameter(msg) from exc

    try:
        prior = next(t for t in study_obj.trials if t.number == trial_id)
    except StopIteration as exc:
        msg = f"trial #{trial_id} not found in study {study_name_arg!r}"
        raise typer.BadParameter(msg) from exc

    # Enqueue the prior params; study.optimize(..., n_trials=1) will pick it up.
    study_obj.enqueue_trial(prior.params)
    objective = build_objective(cfg)
    study_obj.optimize(objective, n_trials=1)
    typer.echo(f"retried trial #{trial_id} as new trial #{len(study_obj.trials) - 1}")
