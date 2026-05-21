"""``rux-ml registry`` verb group — promoted bundles (PR-010 + PR-032 bodies)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig
from rux_ml.registry.champion import read_champion
from rux_ml.registry.paths import champion_path
from rux_ml.registry.promote import promote as do_promote
from rux_ml.registry.promote import rollback as do_rollback
from rux_ml.registry.scorer import score_bundle_on_holdout

app = typer.Typer(
    name="registry",
    no_args_is_help=True,
    help="Filesystem model registry (per-problem promoted bundles).",
)


def _load_cfg(ctx: typer.Context) -> RuxMLConfig:
    opts = get_options(ctx)
    return RuxMLConfig.from_layers(
        opts.config,
        problem=opts.problem,
        study=opts.study,
        overrides=opts.overrides,
    )


@app.command(name="promote")
def promote(
    ctx: typer.Context,
    problem: Annotated[str, typer.Option("--problem", help="Problem name.")],
    study: Annotated[str, typer.Option("--study", help="Originating Optuna study.")],
    trial: Annotated[int, typer.Option("--trial", help="Originating trial number.")],
) -> None:
    """Promote a trial to a new registry version + atomically rewrite champion.json."""
    cfg = _load_cfg(ctx)
    try:
        version = do_promote(cfg, problem=problem, study_name=study, trial_number=trial)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise typer.BadParameter(f"promotion failed: {exc}") from exc

    typer.echo(f"promoted: {problem}@{version}")
    typer.echo(f"  source:  study={study} trial={trial}")
    typer.echo(f"  champion: {champion_path(cfg.registry.root, problem)}")


@app.command(name="list")
def list_(ctx: typer.Context) -> None:
    """List all problems with current champion + recent versions."""
    cfg = _load_cfg(ctx)
    root = cfg.registry.root
    if not root.exists():
        typer.echo(f"no registry at {root}")
        return
    problems = sorted(p for p in root.iterdir() if p.is_dir())
    if not problems:
        typer.echo(f"no problems registered at {root}")
        return
    for problem_path in problems:
        typer.echo(f"\n[{problem_path.name}]")
        champ_file = champion_path(root, problem_path.name)
        if champ_file.exists():
            champ = read_champion(champ_file)
            typer.echo(f"  champion: {champ.get('version')}")
            typer.echo(f"  promoted_at: {champ.get('promoted_at')}")
        else:
            typer.echo("  champion: <none>")
        versions = sorted(
            (p.name for p in problem_path.iterdir() if p.is_dir() and p.name.startswith("v_")),
            reverse=True,
        )
        if versions:
            typer.echo("  versions:")
            for v in versions[:10]:  # most recent 10
                typer.echo(f"    {v}")


@app.command(name="score")
def score(
    ctx: typer.Context,
    problem: Annotated[str, typer.Option("--problem", help="Problem name.")],
    version: Annotated[
        str | None,
        typer.Option("--version", help="Bundle version-id; defaults to current champion."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Receipts directory (created if missing). Default: ./receipts/",
        ),
    ] = Path("receipts"),
) -> None:
    """Score a promoted bundle on its held-out test fold (PR-032).

    Loads the bundle (default: champion), reconstructs ``splits["test"]``
    via ``make_splits`` (deterministic for ``time_ordered``; entropy_hex-
    derived for ``random`` via the originating trial's provenance), runs
    the existing metric registry on a Booster shim, and writes a
    Pydantic-validated JSON receipt + Polars predictions parquet to
    ``output``.

    Naming note: ``score`` differs from MLflow's CLI verbs — MLflow
    ``predict`` is predictions-only; MLflow ``evaluate`` is Python-only.
    rux-ml's ``score`` produces both predictions AND metrics in one CLI
    invocation. Receipt schema body follows MLflow / Kedro convention
    (``metrics`` dict + ``artifacts`` dict).
    """
    cfg = _load_cfg(ctx)
    try:
        receipt = score_bundle_on_holdout(
            cfg, problem=problem, version=version, output_dir=output
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise typer.BadParameter(f"score failed: {exc}") from exc

    typer.echo(f"scored:  {problem}@{receipt.bundle_version}")
    for metric_name, metric_value in receipt.metrics.items():
        typer.echo(f"  {metric_name} (holdout): {metric_value:.6f}")
    typer.echo(
        f"  trial val {receipt.promoted_from.metric_value:.6f} "
        f"(study={receipt.promoted_from.study} trial={receipt.promoted_from.trial_number})"
    )
    holdout = receipt.holdout
    typer.echo(f"  holdout: n_rows={holdout.n_rows} split_kind={holdout.split_kind}")
    if holdout.time_range is not None:
        typer.echo(f"           time_range=[{holdout.time_range[0]}, {holdout.time_range[1]}]")
    for art_name, art in receipt.artifacts.items():
        typer.echo(f"  {art_name}: {art.path}")


@app.command(name="rollback")
def rollback(
    ctx: typer.Context,
    problem: Annotated[str, typer.Option("--problem", help="Problem name.")],
    to: Annotated[
        str, typer.Option("--to", help="Target version-id (e.g. v_2026_05_16_a8f3c2).")
    ],
) -> None:
    """Atomically rewrite champion.json to point at a prior version."""
    cfg = _load_cfg(ctx)
    try:
        do_rollback(cfg, problem=problem, version=to)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    champ_file = champion_path(cfg.registry.root, problem)
    champ = read_champion(champ_file)
    typer.echo(f"rolled back: {problem} champion -> {champ['version']}")
    typer.echo(f"  champion: {champ_file}")
