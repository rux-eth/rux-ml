"""Tests for the per-trial diagnostic artifact upload helpers (per PR-034).

Includes the mandatory BGGC mitigation test
(``test_artifact_upload_roundtrip_via_get_all_artifact_meta``) that bounds the
unverified-synthesis risk Phase 3 Group-D Probe 2 surfaced on the 4-element
combination ``FileSystemArtifactStore(per-study path) + upload_artifact
in-objective + flat metrics.json + get_all_artifact_meta retrieval``. Without
it the BGGC label would be unbounded — the upload + retrieval path could
silently drop fields or scramble payloads and only surface as missing data
under ``rux-ml runs show`` later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna
import pytest
from optuna.artifacts import FileSystemArtifactStore, download_artifact

from rux_ml.config import RuxMLConfig
from rux_ml.runs import (
    FOLD_META_REQUIRED_KEYS,
    build_metrics_dict,
    list_trial_artifacts,
    make_artifact_store,
    upload_diagnostics,
)


def _cfg_with_artifacts_root(tmp_path: Path) -> RuxMLConfig:
    """Build a minimal RuxMLConfig with ``runs.artifacts_root`` pointing under tmp_path."""
    cfg_path = tmp_path / "base.toml"
    cfg_path.write_text(
        f"""
[data]
target_column = "y"

[runs]
storage_url = "sqlite:///{tmp_path}/studies/studies.db"
artifacts_root = "{tmp_path}/studies/artifacts"
"""
    )
    return RuxMLConfig.from_layers(cfg_path)


def _make_study(tmp_path: Path, name: str = "test_study") -> optuna.Study:
    (tmp_path / "studies").mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{tmp_path}/studies/studies.db")
    return optuna.create_study(study_name=name, storage=storage)


# ---------------------------------------------------------------------------
# make_artifact_store
# ---------------------------------------------------------------------------


def test_make_artifact_store_creates_per_study_dir(tmp_path: Path) -> None:
    """Per Optuna 4.8 FAQ: each study gets its own subdirectory under artifacts_root."""
    cfg = _cfg_with_artifacts_root(tmp_path)
    store = make_artifact_store(cfg, study_name="my_study")
    assert isinstance(store, FileSystemArtifactStore)
    expected = tmp_path / "studies" / "artifacts" / "my_study"
    assert expected.exists()
    assert expected.is_dir()


def test_make_artifact_store_isolates_studies(tmp_path: Path) -> None:
    """Distinct studies write into distinct directories (cleanup-friendly per FAQ)."""
    cfg = _cfg_with_artifacts_root(tmp_path)
    make_artifact_store(cfg, study_name="study_a")
    make_artifact_store(cfg, study_name="study_b")
    assert (tmp_path / "studies" / "artifacts" / "study_a").exists()
    assert (tmp_path / "studies" / "artifacts" / "study_b").exists()


# ---------------------------------------------------------------------------
# build_metrics_dict — flat Dict[str, float] convention
# ---------------------------------------------------------------------------


def test_build_metrics_dict_is_flat_dict_str_float() -> None:
    out = build_metrics_dict("rmse", [0.0268, 0.0272, 0.0275], peak_rss_mb=18233.0)
    assert all(isinstance(k, str) for k in out)
    assert all(isinstance(v, float) for v in out.values()), out
    assert out["fold_0_rmse"] == pytest.approx(0.0268)
    assert out["fold_1_rmse"] == pytest.approx(0.0272)
    assert out["fold_2_rmse"] == pytest.approx(0.0275)
    assert out["n_folds"] == 3.0
    assert out["peak_rss_mb"] == pytest.approx(18233.0)
    assert out["metric_mean"] == pytest.approx((0.0268 + 0.0272 + 0.0275) / 3)


def test_build_metrics_dict_handles_single_fold() -> None:
    """Baseline (n_folds=1) has zero std by convention (pstdev of single sample)."""
    out = build_metrics_dict("auc", [0.91], peak_rss_mb=1234.5)
    assert out["fold_0_auc"] == pytest.approx(0.91)
    assert out["metric_mean"] == pytest.approx(0.91)
    assert out["metric_std"] == 0.0  # single-fold convention
    assert out["n_folds"] == 1.0


# ---------------------------------------------------------------------------
# upload_diagnostics + FOLD_META_REQUIRED_KEYS structural guarantee
# ---------------------------------------------------------------------------


def test_upload_diagnostics_persists_both_files_and_returns_ids(tmp_path: Path) -> None:
    cfg = _cfg_with_artifacts_root(tmp_path)
    study = _make_study(tmp_path)
    store = make_artifact_store(cfg, study_name=study.study_name)

    metrics = build_metrics_dict("rmse", [0.5, 0.6], peak_rss_mb=100.0)
    ts_0 = "2026-05-21T00:00:00+00:00"
    ts_1 = "2026-05-21T00:00:01+00:00"
    fold_meta: list[dict[str, Any]] = [
        {"fold_idx": 0, "row_count": 1000, "fit_seconds": 1.5, "timestamp": ts_0},
        {"fold_idx": 1, "row_count": 1000, "fit_seconds": 1.6, "timestamp": ts_1},
    ]

    def objective(trial: optuna.Trial) -> float:
        m_id, f_id = upload_diagnostics(
            trial, store, metrics=metrics, fold_meta=fold_meta, tmp_dir=tmp_path / "scratch"
        )
        assert m_id and f_id and m_id != f_id
        return 0.55

    study.optimize(objective, n_trials=1)

    # Both artifacts land on disk under the per-study root (flat layout by uuid4).
    store_root = tmp_path / "studies" / "artifacts" / study.study_name
    files = sorted(store_root.iterdir())
    assert len(files) == 2


def test_upload_diagnostics_rejects_fold_meta_missing_required_keys(tmp_path: Path) -> None:
    cfg = _cfg_with_artifacts_root(tmp_path)
    study = _make_study(tmp_path)
    store = make_artifact_store(cfg, study_name=study.study_name)

    bad_fold_meta = [{"fold_idx": 0, "row_count": 1000}]  # missing fit_seconds + timestamp

    def objective(trial: optuna.Trial) -> float:
        with pytest.raises(ValueError, match="fold_meta entry missing required keys"):
            upload_diagnostics(
                trial,
                store,
                metrics={"x": 1.0},
                fold_meta=bad_fold_meta,
                tmp_dir=tmp_path / "scratch",
            )
        return 0.0

    study.optimize(objective, n_trials=1)


def test_fold_meta_required_keys_is_locked_set() -> None:
    """Schema is self-checking — any future change must update this test."""
    expected = frozenset({"fold_idx", "row_count", "fit_seconds", "timestamp"})
    assert expected == FOLD_META_REQUIRED_KEYS


# ---------------------------------------------------------------------------
# BGGC mitigation — mandatory round-trip via get_all_artifact_meta
# ---------------------------------------------------------------------------


def test_artifact_upload_roundtrip_via_get_all_artifact_meta(tmp_path: Path) -> None:
    """MANDATORY BGGC mitigation — Phase 3 Group-D Probe 2 NO-cite synthesis.

    Bounds the 4-element combination (`FileSystemArtifactStore(per-study path)`
    + `upload_artifact in-objective` + `flat metrics.json` + `get_all_artifact_meta
    retrieval`). Round-trip: upload → list via Optuna's public API → download
    via `artifact_id` → assert JSON content equality. Without this test the
    BGGC label is unbounded — silent payload drops would only surface
    much later as missing data under `rux-ml runs show`.
    """
    cfg = _cfg_with_artifacts_root(tmp_path)
    study = _make_study(tmp_path, name="bggc_roundtrip")
    store = make_artifact_store(cfg, study_name=study.study_name)

    metrics_payload = build_metrics_dict("rmse", [0.0268, 0.0272], peak_rss_mb=18233.0)
    ts_0 = "2026-05-21T09:42:11+00:00"
    ts_1 = "2026-05-21T09:49:03+00:00"
    fold_meta_payload: list[dict[str, Any]] = [
        {"fold_idx": 0, "row_count": 2070000, "fit_seconds": 412.3, "timestamp": ts_0},
        {"fold_idx": 1, "row_count": 2070000, "fit_seconds": 408.7, "timestamp": ts_1},
    ]

    def objective(trial: optuna.Trial) -> float:
        upload_diagnostics(
            trial,
            store,
            metrics=metrics_payload,
            fold_meta=fold_meta_payload,
            tmp_dir=tmp_path / "scratch",
        )
        return 0.027

    study.optimize(objective, n_trials=1)

    # Retrieve via list_trial_artifacts (which uses get_all_artifact_meta under the hood).
    storage_url = cfg.runs.storage_url
    artifacts = list_trial_artifacts(storage_url, "bggc_roundtrip", trial_number=0)
    assert len(artifacts) == 2
    filenames = {meta.filename for meta in artifacts}
    assert filenames == {"metrics.json", "fold_meta.json"}

    # Round-trip each artifact: download via artifact_id, parse JSON, assert equality.
    download_dir = tmp_path / "downloaded"
    download_dir.mkdir()
    by_filename = {meta.filename: meta for meta in artifacts}

    metrics_path = download_dir / "metrics.json"
    download_artifact(
        artifact_store=store,
        file_path=str(metrics_path),
        artifact_id=by_filename["metrics.json"].artifact_id,
    )
    fold_meta_path = download_dir / "fold_meta.json"
    download_artifact(
        artifact_store=store,
        file_path=str(fold_meta_path),
        artifact_id=by_filename["fold_meta.json"].artifact_id,
    )
    assert json.loads(metrics_path.read_text()) == metrics_payload
    assert json.loads(fold_meta_path.read_text()) == fold_meta_payload


# ---------------------------------------------------------------------------
# list_trial_artifacts
# ---------------------------------------------------------------------------


def test_list_trial_artifacts_returns_empty_when_no_uploads(tmp_path: Path) -> None:
    """Trials that don't reach the upload site (e.g., pruned early) return []."""
    _ = _cfg_with_artifacts_root(tmp_path)
    study = _make_study(tmp_path, name="no_uploads")

    def objective(_trial: optuna.Trial) -> float:
        return 0.5

    study.optimize(objective, n_trials=1)
    artifacts = list_trial_artifacts(
        f"sqlite:///{tmp_path}/studies/studies.db", "no_uploads", trial_number=0
    )
    assert artifacts == []


def test_list_trial_artifacts_raises_for_missing_trial(tmp_path: Path) -> None:
    _ = _cfg_with_artifacts_root(tmp_path)
    study = _make_study(tmp_path, name="missing_trial_test")

    def objective(_trial: optuna.Trial) -> float:
        return 0.5

    study.optimize(objective, n_trials=1)
    with pytest.raises(KeyError, match="trial #99 not found"):
        list_trial_artifacts(
            f"sqlite:///{tmp_path}/studies/studies.db", "missing_trial_test", trial_number=99
        )
