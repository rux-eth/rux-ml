"""Per-trial diagnostic artifact uploads via Optuna's `FileSystemArtifactStore` (per PR-034).

Closes phantom #3 from the 2026-05-20 audit. `RunsConfig.artifacts_root` is
no longer documented-aspiration — it now backs an actual `FileSystemArtifactStore`
that receives a per-trial summary-stats payload split into two files:

- ``metrics.json`` — flat ``Dict[str, float]``: per-fold metric values keyed
  ``fold_<i>_<metric>`` plus aggregates ``metric_mean`` / ``metric_std`` /
  ``n_folds`` / ``peak_rss_mb``. Convention: MLflow ``EvaluationResult.save
  → metrics.json`` at v2.22.4 + Kedro ``MetricsDataset.save`` at
  kedro-datasets-5.1.0 (≥2 cited systems persist a flat dict for metrics).
  ``peak_rss_mb`` is process-wide per-trial (Watchdog measures across all
  folds), so it lives at trial scope, not per-fold.

- ``fold_meta.json`` — mixed-type per-fold sidecar
  (``fold_idx``/``row_count``/``fit_seconds``/``timestamp``).
  Novel to this workbench: ``best-guess-given-constraints`` — no cited
  precedent for the exact shape; bounded by the structural test in
  ``tests/runs/test_artifacts.py`` plus the round-trip mitigation test
  (``test_artifact_upload_roundtrip_via_get_all_artifact_meta``).

**Layout** (per Optuna 4.8 FAQ): one ``FileSystemArtifactStore`` per study, rooted
at ``cfg.runs.artifacts_root / study_name``. Cleanup is then ``rm -rf
studies/artifacts/<study>/`` — Optuna does not ship a delete API and explicitly
declines to support one ("hard to officially support the delete feature and
they are not planning to support this feature in the future" — 4.8.0 FAQ).

**Upload site** (per Q3 convention — Optuna tutorial + ``pytorch_checkpoint.py``
+ ``dashboard/hitl/main.py``): in-objective, post-fit, **no try/except guard**.
Upload happens only on trial success; pruned / memory-pressure trials bubble
exceptions and skip the upload by construction.

**Retrieval**: ``optuna.artifacts.get_all_artifact_meta(trial, storage=storage)``
— Optuna auto-persists each artifact's metadata in
``trial.system_attrs["artifacts:<uuid4>"]``, so ``TrialAttrs`` does NOT need an
``artifact_ids`` field (strict provenance schema stays clean).

All ``upload_artifact`` calls pass kwargs only — Optuna marked the positional
API deprecated at 4.0 and removes it at 6.0.
"""

from __future__ import annotations

import json
import statistics
from typing import TYPE_CHECKING, Any

import optuna
from optuna.artifacts import (
    ArtifactMeta,
    FileSystemArtifactStore,
    get_all_artifact_meta,
    upload_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path

    from rux_ml.config import RuxMLConfig


# Single-sample pstdev is 0 by convention (matches MLflow/Kedro flat-dict shape).
_MIN_FOLDS_FOR_STD: int = 2


# Required keys per entry in ``fold_meta.json``. Codified here so the schema
# stays self-checking (tested in ``tests/runs/test_artifacts.py``).
FOLD_META_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"fold_idx", "row_count", "fit_seconds", "timestamp"}
)


def make_artifact_store(cfg: RuxMLConfig, *, study_name: str) -> FileSystemArtifactStore:
    """Construct a study-scoped ``FileSystemArtifactStore``.

    Per Optuna 4.8 FAQ — separate base directory per study so retention is
    a single ``rm -rf studies/artifacts/<study_name>/`` operation. The parent
    ``cfg.runs.artifacts_root`` and the per-study subdir are both created
    on demand (``parents=True``, ``exist_ok=True``).
    """
    store_root = cfg.runs.artifacts_root / study_name
    store_root.mkdir(parents=True, exist_ok=True)
    return FileSystemArtifactStore(str(store_root))


def build_metrics_dict(
    metric_name: str,
    fold_scores: list[float],
    *,
    peak_rss_mb: float,
) -> dict[str, float]:
    """Flat ``Dict[str, float]`` per MLflow + Kedro convention.

    Per-fold values keyed ``fold_<i>_<metric>``; aggregates as
    ``metric_mean`` / ``metric_std`` / ``n_folds`` / ``peak_rss_mb``.
    The ``fold_<i>_`` prefix idiom is BGGC (no exact precedent surveyed);
    the surrounding shape (flat ``Dict[str, float]``) is cited convention.

    ``peak_rss_mb`` is the Watchdog's process-wide peak across the full
    trial (all folds combined) — placed at trial scope rather than
    per-fold because Watchdog samples globally.
    """
    n = len(fold_scores)
    out: dict[str, float] = {
        f"fold_{i}_{metric_name}": float(score) for i, score in enumerate(fold_scores)
    }
    out["metric_mean"] = float(statistics.fmean(fold_scores)) if n else 0.0
    out["metric_std"] = (
        float(statistics.pstdev(fold_scores)) if n >= _MIN_FOLDS_FOR_STD else 0.0
    )
    out["n_folds"] = float(n)
    out["peak_rss_mb"] = float(peak_rss_mb)
    return out


def upload_diagnostics(
    trial: optuna.Trial,
    artifact_store: FileSystemArtifactStore,
    *,
    metrics: dict[str, float],
    fold_meta: list[dict[str, Any]],
    tmp_dir: Path,
) -> tuple[str, str]:
    """Write + upload ``metrics.json`` and ``fold_meta.json`` for this trial.

    Returns ``(metrics_artifact_id, fold_meta_artifact_id)`` — the IDs are
    auto-persisted by Optuna into ``trial.system_attrs`` so callers do not
    need to record them, but returning them keeps tests + in-process
    inspection straightforward.

    No try/except guard around the uploads — Q3 convention. Failures
    surface as test failures or production exceptions, not silent skips.
    """
    for entry in fold_meta:
        missing = FOLD_META_REQUIRED_KEYS - entry.keys()
        if missing:
            msg = (
                f"fold_meta entry missing required keys: {sorted(missing)} "
                f"(got {sorted(entry.keys())})"
            )
            raise ValueError(msg)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tmp_dir / "metrics.json"
    fold_meta_path = tmp_dir / "fold_meta.json"
    metrics_path.write_text(json.dumps(metrics))
    fold_meta_path.write_text(json.dumps(fold_meta))

    metrics_id = upload_artifact(
        artifact_store=artifact_store,
        file_path=str(metrics_path),
        study_or_trial=trial,
    )
    fold_meta_id = upload_artifact(
        artifact_store=artifact_store,
        file_path=str(fold_meta_path),
        study_or_trial=trial,
    )
    return metrics_id, fold_meta_id


def list_trial_artifacts(
    storage_url: str, study_name: str, trial_number: int
) -> list[ArtifactMeta]:
    """Enumerate ``(artifact_id, filename, mimetype, encoding)`` for one trial.

    Reads Optuna's auto-persisted artifact metadata from
    ``trial.system_attrs["artifacts:<id>"]`` via the public
    :func:`optuna.artifacts.get_all_artifact_meta` API. Returns an empty list
    when the trial has no uploaded artifacts (e.g., legacy pre-PR-034 trials
    or trials that pruned before reaching the upload site).
    """
    storage = optuna.storages.get_storage(storage_url)
    study = optuna.load_study(study_name=study_name, storage=storage)
    try:
        trial = next(t for t in study.trials if t.number == trial_number)
    except StopIteration as exc:
        msg = f"trial #{trial_number} not found in study {study_name!r}"
        raise KeyError(msg) from exc
    return list(get_all_artifact_meta(trial, storage=storage))
