"""``rux-ml data`` verb group — versioning + hashing (real bodies land in PR-004)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rux_ml.cli._shared import get_options, not_implemented

app = typer.Typer(
    name="data",
    no_args_is_help=True,
    help="Dataset versioning + content-addressed hashing.",
)


@app.command(name="hash")
def hash_(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Parquet file or partitioned directory.")],
) -> None:
    """Print the composite ``data_hash`` (bytes_hash + logical_hash) for PATH."""
    _ = get_options(ctx)
    not_implemented(f"data hash {path}", "PR-004")


@app.command(name="version")
def version(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Logical dataset name.")],
    path: Annotated[Path, typer.Argument(help="Source Parquet file or directory.")],
) -> None:
    """Snapshot a dataset into the CAS and write a manifest under that NAME."""
    _ = get_options(ctx)
    not_implemented(f"data version {name} {path}", "PR-004")


@app.command(name="list")
def list_(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Option("--name", help="Filter to one dataset name.")] = None,
) -> None:
    """List versioned datasets from the manifests directory."""
    _ = get_options(ctx)
    filter_clause = f" --name {name}" if name else ""
    not_implemented(f"data list{filter_clause}", "PR-004")
