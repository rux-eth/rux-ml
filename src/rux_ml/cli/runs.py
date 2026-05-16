"""``rux-ml runs`` verb group — Optuna-as-experiment-log queries (real bodies land in PR-009)."""

from __future__ import annotations

from typing import Annotated

import typer

from rux_ml.cli._shared import get_options, not_implemented

app = typer.Typer(
    name="runs",
    no_args_is_help=True,
    help="Query the Optuna-backed experiment log.",
)


@app.command(name="list")
def list_(
    ctx: typer.Context,
    study: Annotated[str | None, typer.Option("--study", help="Filter to one study.")] = None,
    problem: Annotated[str | None, typer.Option("--problem", help="Filter to one problem.")] = None,
) -> None:
    """List trials across studies."""
    _ = get_options(ctx)
    filters: list[str] = []
    if study:
        filters.append(f"--study {study}")
    if problem:
        filters.append(f"--problem {problem}")
    not_implemented(f"runs list {' '.join(filters)}".strip(), "PR-009")


@app.command(name="show")
def show(
    ctx: typer.Context,
    trial_id: Annotated[int, typer.Argument(help="Optuna trial id.")],
) -> None:
    """Display params + metrics + full provenance triple for a trial."""
    _ = get_options(ctx)
    not_implemented(f"runs show {trial_id}", "PR-009")


@app.command(name="compare")
def compare(
    ctx: typer.Context,
    trial_ids: Annotated[list[int], typer.Argument(help="Two or more trial ids.")],
) -> None:
    """Side-by-side params + metrics comparison."""
    _ = get_options(ctx)
    not_implemented(f"runs compare {' '.join(str(t) for t in trial_ids)}", "PR-009")
