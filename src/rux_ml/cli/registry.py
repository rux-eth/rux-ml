"""``rux-ml registry`` verb group — promoted model bundles (real bodies land in PR-010)."""

from __future__ import annotations

from typing import Annotated

import typer

from rux_ml.cli._shared import get_options, not_implemented

app = typer.Typer(
    name="registry",
    no_args_is_help=True,
    help="Filesystem model registry (per-problem promoted bundles).",
)


@app.command(name="promote")
def promote(
    ctx: typer.Context,
    problem: Annotated[str, typer.Option("--problem", help="Problem name.")],
    study: Annotated[str, typer.Option("--study", help="Originating Optuna study.")],
    trial: Annotated[int, typer.Option("--trial", help="Originating trial id.")],
) -> None:
    """Promote a trial to a new registry version + atomically rewrite champion.json."""
    _ = get_options(ctx)
    not_implemented(
        f"registry promote --problem {problem} --study {study} --trial {trial}",
        "PR-010",
    )


@app.command(name="list")
def list_(
    ctx: typer.Context,
    problem: Annotated[str | None, typer.Option("--problem", help="Filter to one problem.")] = None,
) -> None:
    """List problems + current champion + recent versions."""
    _ = get_options(ctx)
    filter_clause = f" --problem {problem}" if problem else ""
    not_implemented(f"registry list{filter_clause}", "PR-010")


@app.command(name="rollback")
def rollback(
    ctx: typer.Context,
    problem: Annotated[str, typer.Option("--problem", help="Problem name.")],
    to_version: Annotated[str, typer.Option("--to", help="Version id to roll back to.")],
) -> None:
    """Atomically rewrite champion.json to point at a prior version."""
    _ = get_options(ctx)
    not_implemented(f"registry rollback --problem {problem} --to {to_version}", "PR-010")
