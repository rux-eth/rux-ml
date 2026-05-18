"""Typer CLI entry point — single ``rux-ml`` binary with nested verb groups.

Per D11 + D15:

    rux-ml [--config PATH] [--problem NAME] [--study NAME]
           [--set key=value ...] [--verbose] [--dry-run]
           <verb> [<sub-verb>] [args...]

Verb groups: ``data``, ``tune``, ``runs``, ``registry``.
Leaf verb:   ``train`` (single baseline run).

The root ``@app.callback()`` parses global flags into a :class:`GlobalOptions`
payload stashed on ``typer.Context.obj``. Subcommands retrieve it via
``rux_ml.cli._shared.get_options(ctx)``. Real bodies for each subcommand land
in their respective PRs (see ``docs/0.0/ROADMAP.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rux_ml import __version__
from rux_ml.cli import data, registry, runs, solve, train, tune
from rux_ml.cli._shared import GlobalOptions, parse_set_overrides

app = typer.Typer(
    name="rux-ml",
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="rux-ml — personal ML research workbench for tabular GBM lifecycle.",
)

# Nested verb groups.
app.add_typer(data.app, name="data")
app.add_typer(tune.app, name="tune")
app.add_typer(runs.app, name="runs")
app.add_typer(registry.app, name="registry")

# Leaf verbs (single command, no sub-verb).
app.command(name="train", help="Run a single baseline training (no sweep).")(train.run_command)
app.command(name="solve", help="Run a single solver invocation (per PR-020).")(solve.run_command)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rux-ml {__version__}")
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Base TOML config path (default: configs/base.toml).",
            show_default=False,
        ),
    ] = Path("configs/base.toml"),
    problem: Annotated[
        str | None,
        typer.Option(
            "--problem",
            "-p",
            help="Name of a configs/problems/<name>.toml overlay.",
        ),
    ] = None,
    study: Annotated[
        str | None,
        typer.Option(
            "--study",
            "-s",
            help="Name of a configs/studies/<name>.toml overlay.",
        ),
    ] = None,
    set_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help=(
                "Override a config field via dot-path, e.g. "
                "--set training.learning_rate=0.05. Values are JSON-parsed when "
                "possible. Repeatable."
            ),
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would happen; don't write artifacts."),
    ] = False,
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """rux-ml — personal ML research workbench."""
    ctx.obj = GlobalOptions(
        config=config,
        problem=problem,
        study=study,
        overrides=parse_set_overrides(list(set_overrides or [])),
        verbose=verbose,
        dry_run=dry_run,
    )


__all__ = ["app", "main"]
