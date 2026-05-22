"""Runs layer — Optuna-as-experiment-log read API + provenance recording.

PR-007 landed the write-side scaffolding (``runs/provenance.py``) so both
``cli/train.py`` (1-trial baseline) and ``cli/tune.py`` (sweep) record the
same per-trial set. PR-009 formalizes the schema (``TrialAttrs`` Pydantic
model), adds the ``one_off_run`` context manager, and lands the query API
that the ``runs`` CLI verb group consumes.
"""

from rux_ml.runs.artifacts import (
    FOLD_META_REQUIRED_KEYS,
    build_metrics_dict,
    list_trial_artifacts,
    make_artifact_store,
    upload_diagnostics,
)
from rux_ml.runs.ask_tell import OneOffRun, one_off_run
from rux_ml.runs.attrs import TrialAttrs
from rux_ml.runs.provenance import (
    HASH_LAYERS,
    data_hashes,
    ensure_storage_parent,
    study_name,
)
from rux_ml.runs.query import Run, compare_runs, list_runs, load_run

__all__ = [
    "FOLD_META_REQUIRED_KEYS",
    "HASH_LAYERS",
    "OneOffRun",
    "Run",
    "TrialAttrs",
    "build_metrics_dict",
    "compare_runs",
    "data_hashes",
    "ensure_storage_parent",
    "list_runs",
    "list_trial_artifacts",
    "load_run",
    "make_artifact_store",
    "one_off_run",
    "study_name",
    "upload_diagnostics",
]
