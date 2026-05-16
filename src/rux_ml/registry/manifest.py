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

from pydantic import BaseModel, ConfigDict, ValidationError


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


class ModelManifest(BaseModel):
    """Promoted model bundle metadata (per D8 + PR-010 sub-decisions C1/D1)."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    version: str  # e.g. "v_2026_05_16_a8f3c2" (D8 BEST-GUESS format)
    problem: str
    promoted_from: PromotedFrom

    # Metric
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
