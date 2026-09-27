"""``rux-ml runs`` verb group — Optuna-as-experiment-log queries (PR-009 bodies)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any

import optuna
import polars as pl
import typer
from optuna.artifacts import download_artifact

from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig
from rux_ml.config.m9_gates import M9GatesError, load_m9_gates
from rux_ml.runs import (
    compare_runs,
    list_runs,
    list_trial_artifacts,
    load_run,
    make_artifact_store,
)
from rux_ml.runs.learning_curve import LearningCurveError, learning_curve_report

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
    study: Annotated[str, typer.Option("--study", "-s", help="Optuna study name (required).")],
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
    else:
        typer.echo("attrs:")
        for key, value in run.attrs.model_dump(exclude_none=True).items():
            typer.echo(f"  {key}: {value}")

    # PR-034: per-trial diagnostic artifacts uploaded by ``tuning/objective.py``
    # and ``cli/train.py``. Read via Optuna's ``get_all_artifact_meta`` — Optuna
    # auto-persists the metadata in ``trial.system_attrs`` so no field on
    # :class:`TrialAttrs` is needed.
    artifacts = list_trial_artifacts(cfg.runs.storage_url, study, trial_number)
    if artifacts:
        typer.echo("diagnostic artifacts:")
        for meta in artifacts:
            typer.echo(f"  - {meta.filename}     (artifact_id: {meta.artifact_id})")


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
    study: Annotated[str, typer.Option("--study", "-s", help="Optuna study name (required).")],
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


def _fold_meta_points(cfg: RuxMLConfig, studies: list[str]) -> list[dict[str, Any]]:
    """Every COMPLETE trial of ``studies`` as a learning-curve point (from fold_meta)."""
    points: list[dict[str, Any]] = []
    storage = cfg.runs.storage_url
    for name in dict.fromkeys(studies):
        try:
            trials = optuna.load_study(study_name=name, storage=storage).get_trials(
                deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
            )
        except KeyError as exc:
            msg = f"no study {name!r} in {storage}"
            raise typer.BadParameter(msg) from exc
        store = make_artifact_store(cfg, study_name=name)
        for trial in trials:
            metas = list_trial_artifacts(storage, name, trial.number)
            ids = [m.artifact_id for m in metas if m.filename == "fold_meta.json"]
            if not ids:
                msg = f"{name} trial {trial.number} has no fold_meta.json"
                raise typer.BadParameter(msg)
            with tempfile.TemporaryDirectory(prefix="rux_ml_lc_") as tmp:
                out = Path(tmp) / "fold_meta.json"
                download_artifact(artifact_store=store, artifact_id=ids[0], file_path=str(out))
                entry = json.loads(out.read_text())[0]
            if "oos" not in entry:
                msg = f"{name} trial {trial.number} is not an [m9] fit (no oos record)"
                raise typer.BadParameter(msg)
            split = entry["split_definition"]
            points.append(
                {
                    "study": name,
                    "trial": trial.number,
                    # a fit without the knob is the full window
                    "train_prefix_frac": split.get("train_prefix_frac", 1.0),
                    "split_kind": split["kind"],
                    "oos": entry["oos"],
                }
            )
    return points


@app.command(name="learning-curve")
def learning_curve(
    ctx: typer.Context,
    study: Annotated[
        list[str],
        typer.Option("--study", "-s", help="A study holding prefix fits (repeatable)."),
    ],
    output: Annotated[Path, typer.Option("--output", help="Where to write the JSON report.")],
) -> None:
    """PR-046: the nested-prefix learning curve and D43's "history-limited" verdict.

    Reads every complete trial of the given studies (one target, one regime, the
    configured prefixes — ``train --set data.train_prefix_frac=<f>``), follows
    ``[m9] learning_curve_metric`` in each fit's ``oos`` record and applies the
    operator-signed ``m9_learning_curve_tolerance_rel`` (read fail-closed).
    """
    cfg = _load_cfg(ctx)
    m9 = cfg.m9
    if (
        m9 is None
        or m9.gates is None
        or m9.learning_curve_fractions is None
        or m9.learning_curve_metric is None
        or m9.learning_curve_direction is None
    ):
        msg = (
            "needs [m9] gates, learning_curve_fractions, learning_curve_metric and "
            "learning_curve_direction (the problem config)"
        )
        raise typer.BadParameter(msg)
    try:
        gates = load_m9_gates(m9.gates)
        report = learning_curve_report(
            _fold_meta_points(cfg, study),
            fractions=m9.learning_curve_fractions,
            metric_path=m9.learning_curve_metric,
            direction=m9.learning_curve_direction,
            tolerance_rel=gates.learning_curve_tolerance_rel,
        )
    except (M9GatesError, LearningCurveError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    report["tolerance_key"] = "m9_learning_curve_tolerance_rel"
    report["gates_sha256"] = gates.sha256
    report["target_column"] = cfg.data.target_column
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for point in report["curve"]:
        typer.echo(f"  {point['train_prefix_frac']:>5}: {point['value']}")
    verdict = "FIRED" if report["history_limited"] else "not fired"
    typer.echo(
        f"history-limited: {verdict} (improvement {report['improvement_rel_last_step']:.4f}"
        f" vs tolerance {report['tolerance_rel']}); report: {output}"
    )
