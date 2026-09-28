"""``rux-ml data`` verb group — versioning + hashing (real bodies from PR-004)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig
from rux_ml.data import OracleQuarantineError, compute_data_hash, list_manifests, snapshot
from rux_ml.data.bridge import BridgeError, check_bridge, training_set_sidecar

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
    cfg = _load_cfg(ctx)  # PR-040: [data.oracle] drives the quarantine check
    try:
        result = compute_data_hash(path, oracle=cfg.data.oracle)
    except OracleQuarantineError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command(name="version")
def version(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Logical dataset name.")],
    path: Annotated[Path, typer.Argument(help="Source Parquet file or directory.")],
) -> None:
    """Snapshot a dataset into the CAS and write a manifest under that NAME."""
    cfg = _load_cfg(ctx)
    try:
        manifest = snapshot(
            name=name,
            source_path=path,
            cas_root=cfg.data.cas_root,
            manifests_root=cfg.data.manifests_root,
            oracle=cfg.data.oracle,
        )
    except OracleQuarantineError as exc:
        raise typer.BadParameter(str(exc)) from exc
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


@app.command(name="bridge")
def bridge(
    ctx: typer.Context,
    output: Annotated[Path, typer.Option("--output", help="Where to write rux-ml's sidecar.")],
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            help="The harness set manifest, <set>/manifest.json (equality test; PR-052).",
        ),
    ] = None,
) -> None:
    """PR-047: rux-ml's data-hash sidecar for ``data.source_path`` (one training subtree).

    Writes ``data_hash``, every file with its sha256, the rux-ml commit SHA and the
    Polars version (program PR-024 A13); refuses a source whose loaded files differ
    from the hashed ones. With ``--manifest`` (the harness set manifest), also runs the
    equality test — the subtree's files and the ``data_hash`` the harness lists under
    ``ruxml_sidecars.<subtree>`` — and records it; any difference, or no listed
    ``data_hash``, exits 2 (PR-052).
    """
    cfg = _load_cfg(ctx)
    if cfg.data.source_path is None:
        msg = "data.source_path is required (the training subtree)"
        raise typer.BadParameter(msg)
    try:
        record = training_set_sidecar(cfg.data.source_path, oracle=cfg.data.oracle)
    except (OracleQuarantineError, BridgeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    error: BridgeError | None = None
    if manifest is not None:
        record["manifest_path"] = str(manifest)
        try:
            record["manifest_check"] = check_bridge(
                record, json.loads(manifest.read_text()), subtree=cfg.data.source_path.name
            )
        except BridgeError as exc:
            record["manifest_check"] = {"equal": False, "error": str(exc)}
            error = exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    typer.echo(f"data_hash: {record['data_hash']}  ({len(record['files'])} files)")
    typer.echo(f"sidecar:   {output}")
    if error is not None:
        raise typer.BadParameter(str(error)) from error
