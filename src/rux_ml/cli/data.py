"""``rux-ml data`` verb group — versioning + hashing (real bodies from PR-004)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig
from rux_ml.data import compute_data_hash, list_manifests, snapshot

app = typer.Typer(
    name="data",
    no_args_is_help=True,
    help="Dataset versioning + content-addressed hashing.",
)


def _load_cfg(ctx: typer.Context) -> RuxMLConfig:
    opts = get_options(ctx)
    return RuxMLConfig.from_layers(
        opts.config,
        problem=opts.problem,
        study=opts.study,
        overrides=opts.overrides,
    )


@app.command(name="hash")
def hash_(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Parquet file or partitioned directory.")],
) -> None:
    """Print the composite ``data_hash`` (bytes_hash + logical_hash) for PATH."""
    _ = get_options(ctx)
    result = compute_data_hash(path)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command(name="version")
def version(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Logical dataset name.")],
    path: Annotated[Path, typer.Argument(help="Source Parquet file or directory.")],
) -> None:
    """Snapshot a dataset into the CAS and write a manifest under that NAME."""
    cfg = _load_cfg(ctx)
    manifest = snapshot(
        name=name,
        source_path=path,
        cas_root=cfg.data.cas_root,
        manifests_root=cfg.data.manifests_root,
    )
    typer.echo(f"snapshotted {name} → {manifest.version_id}")
    typer.echo(f"  bytes_hash:   {manifest.bytes_hash}")
    typer.echo(f"  logical_hash: {manifest.logical_hash}")
    typer.echo(f"  row_count:    {manifest.row_count}")
    typer.echo(f"  manifest at:  {cfg.data.manifests_root}/{name}/{manifest.version_id}.json")


@app.command(name="list")
def list_(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Option("--name", help="Filter to one dataset name.")] = None,
) -> None:
    """List versioned datasets from the manifests directory."""
    cfg = _load_cfg(ctx)
    manifests = list_manifests(cfg.data.manifests_root, name=name)
    if not manifests:
        filter_clause = f" with name={name!r}" if name else ""
        typer.echo(f"no manifests found in {cfg.data.manifests_root}{filter_clause}")
        return
    for m in manifests:
        short = m.bytes_hash[:12]
        typer.echo(
            f"{m.name}  {m.version_id}  rows={m.row_count}  bytes_hash={short}…  ({m.created_at})"
        )
