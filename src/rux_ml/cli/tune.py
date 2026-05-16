"""``rux-ml tune`` verb group — Optuna sweeps (real bodies landed in PR-007).

Sequential trials, in-process, SQLite-backed (subprocess-per-trial isolation
lands in PR-008). Per PR-007 Tier-2 research:

- Objective is **K-fold CV-mean** consuming PR-015's Splitter (see
  ``tuning/objective.py``); per-fold scores reported for ``WilcoxonPruner``.
- Sampler default is ``TPESampler``; pruner default is ``WilcoxonPruner``.
- ``BoTorchSampler`` not offered (Optuna 3.6 deprecation); HEBO is opt-in via
  ``optunahub`` + ``hebo`` (sampler factory raises a clear ``ImportError``
  pointing at the install command when selected without those deps).
"""

from __future__ import annotations

from typing import Annotated

import optuna
import typer

from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig
from rux_ml.runs import study_name
from rux_ml.training import optuna_direction
from rux_ml.tuning import build_objective, create_or_load, make_pruner, make_sampler

app = typer.Typer(
    name="tune",
    no_args_is_help=True,
    help="Optuna hyperparameter sweeps.",
)


def _load_cfg(ctx: typer.Context) -> tuple[RuxMLConfig, str | None, str | None]:
    opts = get_options(ctx)
    cfg = RuxMLConfig.from_layers(
        opts.config,
        problem=opts.problem,
        study=opts.study,
        overrides=opts.overrides,
    )
    return cfg, opts.problem, opts.study


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
    cfg, problem, study = _load_cfg(ctx)
    name = _resolve_study_name(study_name_arg, cfg, problem, study)
    study_obj = _open_study(cfg, name)
    typer.echo(f"study:    {name}")
    typer.echo(f"storage:  {cfg.runs.storage_url}")
    typer.echo(f"sampler:  {cfg.tuning.sampler}    pruner: {cfg.tuning.pruner}")
    typer.echo(f"n_trials: {n_trials}")
    objective = build_objective(cfg)
    study_obj.optimize(objective, n_trials=n_trials)
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
    cfg, _, _ = _load_cfg(ctx)
    study_obj = _open_study(cfg, study_name_arg)
    typer.echo(f"resuming study {study_name_arg} ({len(study_obj.trials)} existing trials)")
    objective = build_objective(cfg)
    study_obj.optimize(objective, n_trials=n_trials)
    typer.echo(
        f"\ntotal trials: {len(study_obj.trials)} "
        f"(best {cfg.training.metric}={study_obj.best_value:.6f})"
    )


@app.command(name="status")
def status(
    ctx: typer.Context,
    study_name_arg: Annotated[str, typer.Argument(metavar="STUDY_NAME")],
) -> None:
    """Print progress, current best, and best trial's user_attrs."""
    cfg, _, _ = _load_cfg(ctx)
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
    cfg, _, _ = _load_cfg(ctx)
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
