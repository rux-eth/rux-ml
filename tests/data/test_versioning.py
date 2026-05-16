"""Tests for composite data_hash + manifest + CAS (per D9 + D14)."""

from __future__ import annotations

import errno
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import xxhash

from rux_ml._internal import hashing
from rux_ml.data.loaders import load_parquet, materialize
from rux_ml.data.versioning import (
    compute_data_hash,
    list_manifests,
    read_manifest,
    snapshot,
)


def test_compute_data_hash_returns_all_required_fields(parquet_file: Path) -> None:
    result = compute_data_hash(parquet_file)
    assert set(result) == {"bytes_hash", "logical_hash", "row_count", "schema"}
    assert isinstance(result["bytes_hash"], str)
    assert isinstance(result["logical_hash"], str)
    assert len(result["bytes_hash"]) == 64  # sha256 hex
    assert len(result["logical_hash"]) == 64
    assert result["row_count"] == 10
    assert set(result["schema"]) == {"x1", "x2", "cat", "y"}


def test_bytes_hash_is_stable_across_calls(parquet_file: Path) -> None:
    a = compute_data_hash(parquet_file)["bytes_hash"]
    b = compute_data_hash(parquet_file)["bytes_hash"]
    assert a == b


def test_logical_hash_is_stable_across_calls(parquet_file: Path) -> None:
    a = compute_data_hash(parquet_file)["logical_hash"]
    b = compute_data_hash(parquet_file)["logical_hash"]
    assert a == b


def test_logical_hash_invariant_to_row_order(
    parquet_file: Path, shuffled_parquet_file: Path
) -> None:
    """Same logical rows, different physical order → same logical_hash, different bytes_hash."""
    a = compute_data_hash(parquet_file)
    b = compute_data_hash(shuffled_parquet_file)
    assert a["logical_hash"] == b["logical_hash"]
    assert a["bytes_hash"] != b["bytes_hash"]
    assert a["row_count"] == b["row_count"]


def test_snapshot_writes_manifest_and_hardlinks(
    parquet_file: Path, cas_root: Path, manifests_root: Path
) -> None:
    manifest = snapshot("tiny", parquet_file, cas_root=cas_root, manifests_root=manifests_root)
    # Manifest file lives at manifests/<name>/<version_id>.json
    manifest_path = manifests_root / "tiny" / f"{manifest.version_id}.json"
    assert manifest_path.exists()
    # CAS has the file
    assert len(manifest.cas_files) == 1
    assert Path(manifest.cas_files[0]).exists()
    # Round-trip the manifest
    loaded = read_manifest(manifest_path)
    assert loaded.bytes_hash == manifest.bytes_hash
    assert loaded.logical_hash == manifest.logical_hash
    assert loaded.row_count == 10


def test_snapshot_is_idempotent_for_identical_content(
    parquet_file: Path, cas_root: Path, manifests_root: Path
) -> None:
    a = snapshot("tiny", parquet_file, cas_root=cas_root, manifests_root=manifests_root)
    b = snapshot("tiny", parquet_file, cas_root=cas_root, manifests_root=manifests_root)
    assert a.version_id == b.version_id
    assert a.bytes_hash == b.bytes_hash


def test_list_manifests_filters_by_name(
    parquet_file: Path,
    shuffled_parquet_file: Path,
    cas_root: Path,
    manifests_root: Path,
) -> None:
    snapshot("alpha", parquet_file, cas_root=cas_root, manifests_root=manifests_root)
    snapshot("beta", shuffled_parquet_file, cas_root=cas_root, manifests_root=manifests_root)
    all_m = list_manifests(manifests_root)
    assert len(all_m) == 2
    only_alpha = list_manifests(manifests_root, name="alpha")
    assert len(only_alpha) == 1
    assert only_alpha[0].name == "alpha"


def test_list_manifests_on_missing_root_returns_empty(tmp_path: Path) -> None:
    assert list_manifests(tmp_path / "nope") == []


def test_link_falls_back_to_copy_on_exdev(
    parquet_file: Path, cas_root: Path, manifests_root: Path
) -> None:
    """When os.link raises EXDEV, snapshot must copy instead and warn."""

    def fake_link(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    with (
        patch("rux_ml.data.versioning.os.link", side_effect=fake_link),
        pytest.warns(UserWarning, match="cross-device"),
    ):
        m = snapshot("tiny", parquet_file, cas_root=cas_root, manifests_root=manifests_root)
    # The CAS file exists (via copy fallback)
    assert Path(m.cas_files[0]).exists()


def test_compute_data_hash_works_on_partitioned_dir(parquet_dir: Path) -> None:
    """Hashing must work on hive-style partitioned directories too."""
    result = compute_data_hash(parquet_dir)
    assert result["row_count"] == 10
    assert len(result["bytes_hash"]) == 64
    assert len(result["logical_hash"]) == 64


def test_partitioned_and_unpartitioned_share_logical_hash(
    parquet_file: Path, parquet_dir: Path
) -> None:
    """Same logical data, different physical layout (file vs 2-part dir) → same logical_hash."""
    single = compute_data_hash(parquet_file)
    multi = compute_data_hash(parquet_dir)
    assert single["logical_hash"] == multi["logical_hash"]
    assert single["row_count"] == multi["row_count"]
    # bytes_hashes differ (different file count, different per-file bytes)
    assert single["bytes_hash"] != multi["bytes_hash"]


def test_xxhash_is_python_binding_not_rust(
    parquet_file: Path,
) -> None:
    """Per D13: use mature Python bindings before reaching for custom Rust."""
    _ = parquet_file
    assert hasattr(xxhash, "xxh3_64")
    # And our hashing module uses it (no custom Rust crate)
    assert hashing.xxh3_64_file.__module__ == "rux_ml._internal.hashing"


def test_polars_dataframes_are_returned(tmp_path: Path, tiny_df: pl.DataFrame) -> None:
    """Smoke check: lazy/eager Polars APIs still work for our patterns."""
    p = tmp_path / "x.parquet"
    tiny_df.write_parquet(p)
    lf = load_parquet(p)
    df = materialize(lf)
    assert df.height == 10
