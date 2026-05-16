# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Query helpers over the Optuna-backed experiment log (per PR-009).

The file-level ``pyright`` pragma relaxes the ``reportUnknown*`` family
because ``optuna.study.Study.trials_dataframe()`` returns an
``Unknown``-typed pandas DataFrame as of Optuna 4.8 (Optuna doesn't ship
type stubs for the dataframe shape). Conversion to Polars (which IS typed)
happens immediately at the boundary; library code elsewhere in ``src/``
stays strict.

Three public surfaces:

- :func:`list_runs` returns a Polars DataFrame of trials across studies
  (optionally filtered by study name or problem-prefix).
- :func:`load_run` loads one trial by ``(study_name, trial_number)``, returns
  a :class:`Run` rich object exposing params, value, and validated
  ``TrialAttrs``. Used by ``rux-ml runs show``.
- :func:`compare_runs` returns a wide Polars DataFrame with params + metric +
  cv/training cfg hashes side-by-side. Pipeable per the PR-009 notes.

Trial-identification convention: ``trial.number`` scoped to ``--study NAME``
(per PR-007's ``tune retry-trial`` precedent — sub-decision E1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import optuna
import polars as pl

from rux_ml.runs.attrs import TrialAttrs
from rux_ml.runs.provenance import ensure_storage_parent

if TYPE_CHECKING:
    from collections.abc import Iterable

# Columns from ``study.trials_dataframe()`` we surface for ``runs list``.
_LIST_COLUMNS_BASE: tuple[str, ...] = ("number", "state", "value")


@dataclass
class Run:
    """A single trial's view — params, metric value, validated TrialAttrs.

    ``attrs`` is ``None`` if the trial's ``user_attrs`` don't validate against
    the schema (legacy trials, or in-flight). Promotion (PR-010) refuses to
    promote trials whose ``attrs`` are ``None``.
    """

    study_name: str
    trial_number: int
    state: str
    value: float | None
    params: dict[str, Any]
    attrs: TrialAttrs | None


def _open_study(study_name: str, storage_url: str) -> optuna.Study:
    ensure_storage_parent(storage_url)
    try:
        return optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError as exc:
        msg = f"study {study_name!r} not found at {storage_url}"
        raise KeyError(msg) from exc


def _iter_studies(
    storage_url: str, study: str | None, problem: str | None
) -> Iterable[optuna.Study]:
    """Yield studies in ``storage_url`` matching the optional filters.

    ``problem`` matches against the study-name prefix per the
    ``study_name_template = "{problem}_{study}_{stamp}"`` convention.
    """
    ensure_storage_parent(storage_url)
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    for s in summaries:
        name = s.study_name
        if study is not None and name != study:
            continue
        if problem is not None and not name.startswith(f"{problem}_"):
            continue
        yield optuna.load_study(study_name=name, storage=storage_url)


def list_runs(
    storage_url: str,
    *,
    study: str | None = None,
    problem: str | None = None,
) -> pl.DataFrame:
    """List trials across studies in the given storage as a Polars DataFrame.

    Output columns (always present): ``study_name``, ``trial_number``,
    ``state``, ``value``, plus every ``user_attrs_<key>`` Optuna's
    ``trials_dataframe()`` surfaces (typically the 13 required-now fields
    plus any Optional ones written).
    """
    frames: list[pl.DataFrame] = []
    for s in _iter_studies(storage_url, study=study, problem=problem):
        df = s.trials_dataframe()
        if df.empty:
            continue
        # ``trials_dataframe()`` is pandas-typed; convert to Polars at the boundary.
        pl_df = pl.from_pandas(df)
        pl_df = pl_df.with_columns(pl.lit(s.study_name).alias("study_name")).rename(
            {"number": "trial_number"}
        )
        # Keep base columns first; preserve user_attrs / params columns alongside.
        ordered = ["study_name", "trial_number", "state", "value", *[
            c for c in pl_df.columns
            if c not in {"study_name", "trial_number", "state", "value"}
        ]]
        frames.append(pl_df.select(ordered))
    if not frames:
        return pl.DataFrame(
            schema={
                "study_name": pl.String,
                "trial_number": pl.Int64,
                "state": pl.String,
                "value": pl.Float64,
            }
        )
    return pl.concat(frames, how="diagonal_relaxed")


def load_run(storage_url: str, study_name: str, trial_number: int) -> Run:
    """Load one trial by ``(study_name, trial_number)`` and validate its TrialAttrs.

    Raises ``KeyError`` if the study doesn't exist or the trial isn't in it.
    ``Run.attrs`` is ``None`` when the trial's ``user_attrs`` don't validate
    against :class:`TrialAttrs` (e.g., legacy trials or in-flight trials
    that didn't finish recording).
    """
    study = _open_study(study_name, storage_url)
    try:
        trial = next(t for t in study.trials if t.number == trial_number)
    except StopIteration as exc:
        msg = f"trial #{trial_number} not found in study {study_name!r}"
        raise KeyError(msg) from exc

    attrs: TrialAttrs | None
    try:
        attrs = TrialAttrs.from_trial(trial)
    except Exception:
        attrs = None

    return Run(
        study_name=study_name,
        trial_number=trial.number,
        state=trial.state.name,
        value=trial.value,
        params=dict(trial.params),
        attrs=attrs,
    )


def compare_runs(
    storage_url: str, study_name: str, trial_numbers: list[int]
) -> pl.DataFrame:
    """Side-by-side comparison of multiple trials in one study.

    Each row is one trial; columns include ``trial_number``, ``state``,
    ``value``, every distinct ``params_<key>`` across the trials, plus
    selected ``user_attrs`` rows (the per-layer ``*_cfg_hash`` set is the
    most useful signal — easy to spot which trial differed how).
    """
    rows: list[dict[str, Any]] = []
    for trial_number in trial_numbers:
        run = load_run(storage_url, study_name, trial_number)
        row: dict[str, Any] = {
            "trial_number": run.trial_number,
            "state": run.state,
            "value": run.value,
        }
        for k, v in run.params.items():
            row[f"params_{k}"] = v
        if run.attrs is not None:
            # Surface the per-layer hashes (most diff-friendly) + metric.
            dumped = run.attrs.model_dump(exclude_none=True)
            for key, value in dumped.items():
                if key.endswith("_cfg_hash") or key in {"metric", "best_iteration"}:
                    row[f"attrs_{key}"] = value
        rows.append(row)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)
