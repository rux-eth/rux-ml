"""Pydantic-validated model manifest schema (per D8 + PR-010).

The manifest is a JSON file alongside ``pipeline.skops`` and ``model.ubj`` in
each ``registry/<problem>/<version>/`` directory. It captures the full
provenance triple from the originating Optuna trial plus the library
versions used at promote time so the bundle is reproducible.

``read()`` validates strictly on load; schema mismatches raise
:class:`ManifestSchemaError`. ``write()`` uses tmp + ``os.replace`` for
atomic write so partial files never leak.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class ManifestSchemaError(ValueError):
    """Raised when a ``manifest.json`` fails Pydantic validation on read.

    PR-010's promote refuses to promote when the originating ``TrialAttrs``
    doesn't validate; this error covers the symmetric case at load time.
    """


class PromotedFrom(BaseModel):
    """Where this bundle came from."""

    model_config = ConfigDict(extra="forbid")

    study: str
    trial_number: int
    metric_value: float


class LibraryVersions(BaseModel):
    """Library / runtime versions at promote time."""

    model_config = ConfigDict(extra="forbid")

    xgboost: str
    skops: str
    scikit_learn: str
    polars: str
    numpy: str
    rux_ml: str


class HarnessManifestRef(BaseModel):
    """The harness training set the model was trained on (PR-051; rux-capital program
    v0.3 D45 #6, ACCEPTANCE C11: "rux-ml's model manifest carries the harness manifest
    id"). Read at promote from the view program PR-024's materializer writes beside the
    subtree (``<set>/<subtree>.manifest.json``), and checked against the set manifest."""

    model_config = ConfigDict(extra="forbid")

    # sha256 of the set's ``manifest.json`` — the id the harness's own view cites
    # (``set_manifest_sha256``), verified against the file at promote.
    manifest_id: str
    set_name: str  # the set manifest's ``set`` (its directory name)
    subtree: str  # ``fill`` / ``walk`` — the last part of ``data.source_path``
    view_sha256: str  # sha256 of ``<subtree>.manifest.json`` as read at promote
    # The set manifest's copy of rux-ml's ``data_hash`` (``ruxml_sidecars.<subtree>``);
    # equal to ``ModelManifest.data_hash`` when present, else promote refuses.
    ruxml_data_hash: str | None


class RefitRecord(BaseModel):
    """How promote re-fitted the bundle's booster (PR-059; program PR-028a Q16 / Q4e): what a
    consumer needs to predict with the model that was evaluated.

    ``iteration_range`` is the ``[start, end)`` boosting-round range the trial predicted with:
    ``(0, best_iteration + 1)`` when early stopping ran, else XGBoost's ``(0, 0)`` (every
    tree) — as ``XGBModel._get_iteration_range`` and PR-054's ``fit_batches`` derive it.
    ``best_iteration`` survives ``model.ubj``, but XGBoost 3.2.0's ``Booster.predict``
    default uses every tree (program PR-028a DG-2: up to 1.05 bp off the evaluated walk
    model), so a consumer passes this range. ``device`` / ``nthread`` are the fit's
    (``booster_threads_device``): a loaded model reports ``cpu``, so this is their record.
    """

    model_config = ConfigDict(extra="forbid")

    train_rows: int  # the re-fit's train partition (the trial's ``train_rows``)
    val_rows: int  # its early-stopping partition
    best_iteration: int | None
    iteration_range: tuple[int, int]
    device: str
    nthread: int  # 0 = the OpenMP default

    @model_validator(mode="after")
    def _range_is_the_best_iteration_plus_one(self) -> RefitRecord:
        want = (0, 0) if self.best_iteration is None else (0, self.best_iteration + 1)
        if self.iteration_range != want:
            msg = (
                f"iteration_range {self.iteration_range} is not {want}, the range "
                f"best_iteration={self.best_iteration} predicts with"
            )
            raise ValueError(msg)
        return self


class OosRecord(BaseModel):
    """The trial's out-of-sample record, as its ``fold_meta.json`` ``oos`` carries it
    (PR-059; program PR-028a Phase 4 item 3): the TEST partition's score and, for an
    honesty problem, the honesty test. ``ModelManifest.metric_value`` is the trial's Optuna
    value — the VAL score for an ``[m9]`` fit — so the TEST numbers the program cites are
    these. Carried as written: keys this block does not name (``gates_sha256``,
    ``calibration``, ...) are kept verbatim; a named key the record lacks reads ``None``.
    """

    model_config = ConfigDict(extra="allow")

    partition: str
    n_rows: int
    n_scored: int | None = None
    metric: str | None = None
    score: float | None = None
    honesty: dict[str, Any] | None = None


class ModelManifest(BaseModel):
    """Promoted model bundle metadata (per D8 + PR-010 sub-decisions C1/D1)."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    version: str  # e.g. "v_2026_05_16_a8f3c2" (D8 BEST-GUESS format)
    problem: str
    promoted_from: PromotedFrom

    # Metric — the trial's Optuna value: the VAL score for an [m9] fit (TEST is ``oos``).
    metric_name: str
    metric_value: float

    # 8 per-layer config hashes from the originating ``TrialAttrs`` (PR-009).
    data_cfg_hash: str
    features_cfg_hash: str
    training_cfg_hash: str
    tuning_cfg_hash: str
    runs_cfg_hash: str
    registry_cfg_hash: str
    memory_cfg_hash: str
    cv_cfg_hash: str

    # Root config + git + data provenance.
    root_cfg_hash: str
    git_sha: str
    data_hash: str
    data_bytes_hash: str
    data_logical_hash: str
    # PR-051: the harness set's provenance manifest; None for a source with no harness view.
    harness_manifest: HarnessManifestRef | None = None

    # PR-059 (program PR-028a Q16 / Q4e): how promote re-fitted the booster, the trial's
    # out-of-sample export sha256 (PR-055) and its TEST record. None in a manifest written
    # before them; ``oos_export_sha256`` / ``oos`` None for a trial that recorded none.
    refit: RefitRecord | None = None
    oos_export_sha256: str | None = None
    oos: OosRecord | None = None

    # Library + runtime + features.
    library_versions: LibraryVersions
    feature_list_hash: str

    # Audit
    created_at: str  # ISO 8601 UTC, e.g. "2026-05-16T15:00:00+00:00"


def read(path: Path) -> ModelManifest:
    """Load + validate a ``manifest.json``.

    Raises :class:`ManifestSchemaError` on schema mismatch; the original
    ``pydantic.ValidationError`` is chained as ``__cause__``.
    """
    raw = json.loads(Path(path).read_text())
    try:
        return ModelManifest.model_validate(raw)
    except ValidationError as exc:
        msg = f"manifest at {path} failed schema validation"
        raise ManifestSchemaError(msg) from exc


def write(path: Path, manifest: ModelManifest) -> None:
    """Atomically write ``manifest.json`` via tmp + ``os.replace``.

    The tmp lands in the destination's parent dir (POSIX requires same-FS
    rename for atomicity). Concurrent readers see either the prior file or
    the new one — never a partial write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = manifest.model_dump(mode="json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
