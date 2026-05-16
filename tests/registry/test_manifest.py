"""Tests for ``rux_ml.registry.manifest`` (per PR-010)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rux_ml.registry.manifest import (
    LibraryVersions,
    ManifestSchemaError,
    ModelManifest,
    PromotedFrom,
    read,
    write,
)


def _sample_manifest(**overrides: object) -> ModelManifest:
    base: dict[str, object] = {
        "version": "v_2026_05_16_a8f3c2",
        "problem": "churn_v1",
        "promoted_from": PromotedFrom(study="s1", trial_number=3, metric_value=0.91),
        "metric_name": "auc",
        "metric_value": 0.91,
        "data_cfg_hash": "d" * 64,
        "features_cfg_hash": "f" * 64,
        "training_cfg_hash": "t" * 64,
        "tuning_cfg_hash": "u" * 64,
        "runs_cfg_hash": "r" * 64,
        "registry_cfg_hash": "g" * 64,
        "memory_cfg_hash": "m" * 64,
        "cv_cfg_hash": "c" * 64,
        "root_cfg_hash": "0" * 64,
        "git_sha": "abc1234",
        "data_hash": "bytes_hash|logical_hash",
        "data_bytes_hash": "b" * 64,
        "data_logical_hash": "l" * 64,
        "library_versions": LibraryVersions(
            xgboost="3.2.0",
            skops="0.14.0",
            scikit_learn="1.8.0",
            polars="1.40.1",
            numpy="2.4.5",
            rux_ml="0.0.1",
        ),
        "feature_list_hash": "e" * 64,
        "created_at": "2026-05-16T15:00:00+00:00",
    }
    base.update(overrides)
    return ModelManifest.model_validate(base)


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    original = _sample_manifest()
    write(path, original)
    loaded = read(path)
    assert loaded == original


def test_atomic_write_no_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``os.replace`` raises, the partial tmp must NOT clobber the final path."""
    import os  # noqa: PLC0415

    path = tmp_path / "manifest.json"
    # Seed an existing valid manifest first.
    write(path, _sample_manifest(version="v_2026_05_15_baseline"))
    pre_contents = path.read_text()

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", _raise)
    with pytest.raises(OSError, match="simulated"):
        write(path, _sample_manifest(version="v_2026_05_16_new"))

    # Final path is untouched — concurrent readers still see the pre-failure manifest.
    assert path.read_text() == pre_contents


def test_read_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = _sample_manifest().model_dump(mode="json")
    payload["unknown_field"] = "should_fail"
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestSchemaError):
        read(path)


def test_read_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = _sample_manifest().model_dump(mode="json")
    del payload["data_hash"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestSchemaError):
        read(path)
