"""``rux-ml tune`` verb group — Optuna sweeps (real bodies land in PR-007/PR-008)."""

from __future__ import annotations

from typing import Annotated

import typer

from rux_ml.cli._shared import get_options, not_implemented

app = typer.Typer(
    name="tune",
    no_args_is_help=True,
    help="Optuna hyperparameter sweeps.",
)


@app.command(name="start")
def start(
    ctx: typer.Context,
    study_name: Annotated[str, typer.Argument(help="Study name (creates new).")],
    n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50,
) -> None:
    """Create or load a study and run N trials."""
    _ = get_options(ctx)
    not_implemented(f"tune start {study_name} --n-trials {n_trials}", "PR-007")


@app.command(name="resume")
def resume(
    ctx: typer.Context,
    study_name: Annotated[str, typer.Argument(help="Existing study name.")],
    n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50,
) -> None:
    """Add N more trials to an existing study."""
    _ = get_options(ctx)
    not_implemented(f"tune resume {study_name} --n-trials {n_trials}", "PR-007")


@app.command(name="status")
def status(
    ctx: typer.Context,
    study_name: Annotated[str, typer.Argument(help="Study name.")],
) -> None:
    """Print progress, current best, and best trial's user_attrs."""
    _ = get_options(ctx)
    not_implemented(f"tune status {study_name}", "PR-007")


@app.command(name="retry-trial")
def retry_trial(
    ctx: typer.Context,
    study_name: Annotated[str, typer.Argument()],
    trial_id: Annotated[int, typer.Argument()],
) -> None:
    """Re-run a failed trial with its original params."""
    _ = get_options(ctx)
    not_implemented(f"tune retry-trial {study_name} {trial_id}", "PR-007")
