"""``rux-ml runs`` verb group — Optuna-as-experiment-log queries (PR-009 bodies)."""

from __future__ import annotations

from typing import Annotated

import polars as pl
import typer

from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig
from rux_ml.runs import compare_runs, list_runs, load_run

app = typer.Typer(
    name="runs",
    no_args_is_help=True,
    help="Query the Optuna-backed experiment log.",
)


def _load_cfg(ctx: typer.Context) -> RuxMLConfig:
    opts = get_options(ctx)
    return RuxMLConfig.from_layers(
        opts.config,
        problem=opts.problem,
        study=opts.study,
        overrides=opts.overrides,
    )


@app.command(name="list")
def list_(
    ctx: typer.Context,
    study: Annotated[
        str | None, typer.Option("--study", help="Filter to one study by exact name.")
    ] = None,
    problem: Annotated[
        str | None,
        typer.Option(
            "--problem",
            help="Filter to studies whose name starts with '<problem>_' (template prefix).",
        ),
    ] = None,
) -> None:
    """List trials across studies in the configured Optuna storage."""
    cfg = _load_cfg(ctx)
    df = list_runs(cfg.runs.storage_url, study=study, problem=problem)
    if df.is_empty():
        typer.echo("no trials found")
        return
    # Print as Polars-native table; widen string/cell limits so long names (study,
    # hashes) aren't truncated to ``…``.
    with pl.Config(
        tbl_rows=df.height,
        tbl_cols=df.width,
        fmt_str_lengths=120,
        tbl_width_chars=-1,  # no table-width truncation; output is pipeable per PR-009 notes
    ):
        typer.echo(str(df))


@app.command(name="show")
def show(
    ctx: typer.Context,
    trial_number: Annotated[
        int, typer.Argument(help="Trial number within --study (Optuna trial.number).")
    ],
    study: Annotated[
        str, typer.Option("--study", "-s", help="Optuna study name (required).")
    ],
) -> None:
    """Display one trial's params + metric + 8-layer provenance triple."""
    cfg = _load_cfg(ctx)
    try:
        run = load_run(cfg.runs.storage_url, study, trial_number)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"study:        {run.study_name}")
    typer.echo(f"trial number: {run.trial_number}")
    typer.echo(f"state:        {run.state}")
    typer.echo(f"value:        {run.value}")
    typer.echo("params:")
    for key, value in sorted(run.params.items()):
        typer.echo(f"  {key}: {value}")
    if run.attrs is None:
        typer.echo("attrs:        <missing or invalid — schema validation failed>")
        return
    typer.echo("attrs:")
    for key, value in run.attrs.model_dump(exclude_none=True).items():
        typer.echo(f"  {key}: {value}")


@app.command(name="compare")
def compare(
    ctx: typer.Context,
    trial_numbers: Annotated[
        list[int],
        typer.Argument(
            metavar="TRIAL_NUMBERS",
            help="Two or more trial numbers within --study.",
        ),
    ],
    study: Annotated[
        str, typer.Option("--study", "-s", help="Optuna study name (required).")
    ],
) -> None:
    """Side-by-side params + metric + cfg-hash comparison across trials."""
    cfg = _load_cfg(ctx)
    try:
        df = compare_runs(cfg.runs.storage_url, study, trial_numbers)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    with pl.Config(
        tbl_rows=df.height,
        tbl_cols=df.width,
        fmt_str_lengths=120,
        tbl_width_chars=-1,  # no table-width truncation; output is pipeable per PR-009 notes
    ):
        typer.echo(str(df))
