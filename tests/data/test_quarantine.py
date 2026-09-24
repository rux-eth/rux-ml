"""Oracle quarantine at ingest (PR-040; program ACCEPTANCE C13).

Every refusal asserts :class:`OracleQuarantineError` specifically — never a bare
``Exception`` — because polars already raises unnamed errors on some of these
inputs (a non-empty tag inside a directory source, a heterogeneous directory),
and an any-exception assertion would pass without the quarantine existing.
Tag fixtures are non-empty unless the case is about zero-byte tags (polars
skips zero-byte files silently).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pytest

from rux_ml.config.data import OracleQuarantineConfig
from rux_ml.data import OracleQuarantineError, check_oracle_quarantine
from rux_ml.data.loaders import load_parquet, materialize
from rux_ml.data.versioning import compute_data_hash, snapshot
from rux_ml.runs import data_hashes

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "fixtures" / "golden_v1"


def _clean(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).write_parquet(path)
    return path


def _tag(directory: Path, tag_file: str, *, empty: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / tag_file).write_text("" if empty else '{"oracle": true}')


@pytest.fixture
def ns(oracle_cfg: OracleQuarantineConfig) -> str:
    return oracle_cfg.namespace


@pytest.fixture
def tag(oracle_cfg: OracleQuarantineConfig) -> str:
    return oracle_cfg.tag_file


def _refused(path: Path, cfg: OracleQuarantineConfig | None) -> str:
    with pytest.raises(OracleQuarantineError) as exc:
        check_oracle_quarantine(path, cfg)
    return str(exc.value)


# ---------------------------------------------------------------------------
# Namespace refusals
# ---------------------------------------------------------------------------


def test_top_level_oracle_column_is_refused_and_named(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, ns: str
) -> None:
    p = tmp_path / "t.parquet"
    pl.DataFrame({"x": [1], f"{ns}label": [2]}).write_parquet(p)
    assert f"{ns}label" in _refused(p, oracle_cfg)


@pytest.mark.parametrize("shape", ["struct", "list_struct", "array_struct"])
def test_nested_oracle_field_is_refused_and_path_named(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, ns: str, shape: str
) -> None:
    inner = {"a": 1, f"{ns}z": 2}
    if shape == "struct":
        df = pl.DataFrame({"s": [inner]})
    elif shape == "list_struct":
        df = pl.DataFrame({"s": [[inner]]})
    else:
        df = pl.DataFrame({"s": [[inner]]}).with_columns(
            pl.col("s").cast(pl.Array(pl.Struct({"a": pl.Int64, f"{ns}z": pl.Int64}), 1))
        )
    p = tmp_path / "t.parquet"
    df.write_parquet(p)
    assert f"{ns}z" in _refused(p, oracle_cfg)


@pytest.mark.parametrize("transform", [str.upper, str.title])
def test_case_variant_prefix_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, ns: str, transform: object
) -> None:
    col = f"{transform(ns)}label"  # type: ignore[operator]
    p = tmp_path / "t.parquet"
    pl.DataFrame({"x": [1], col: [2]}).write_parquet(p)
    assert col in _refused(p, oracle_cfg)


def test_hive_partition_key_in_namespace_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, ns: str
) -> None:
    root = tmp_path / "hive"
    _clean(root / f"{ns}flag=1" / "part.parquet")
    assert f"{ns}flag" in _refused(root, oracle_cfg)


def test_heterogeneous_directory_is_refused_by_name_not_schema_error(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, ns: str
) -> None:
    root = tmp_path / "parts"
    root.mkdir()
    pl.DataFrame({"x": [1]}).write_parquet(root / "1.parquet")
    pl.DataFrame({"x": [2], f"{ns}y": [3]}).write_parquet(root / "2.parquet")
    assert f"{ns}y" in _refused(root, oracle_cfg)


def test_literal_bracket_filename_is_read_not_globbed(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, ns: str
) -> None:
    root = tmp_path / "br"
    root.mkdir()
    pl.DataFrame({"x": [1], f"{ns}y": [3]}).write_parquet(root / "p[1].parquet")
    assert f"{ns}y" in _refused(root, oracle_cfg)


# ---------------------------------------------------------------------------
# Tag refusals
# ---------------------------------------------------------------------------


def test_tag_in_the_files_own_directory_is_refused_and_named(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    day = tmp_path / "store" / "day"
    p = _clean(day / "t.parquet")
    _tag(day, tag)
    assert str(day) in _refused(p, oracle_cfg)


def test_tag_several_ancestor_levels_up_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    _tag(tmp_path / "store", tag)
    p = _clean(tmp_path / "store" / "a" / "b" / "c" / "d" / "t.parquet")
    assert str(tmp_path / "store") in _refused(p, oracle_cfg)


def test_tag_on_the_directory_source_itself_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    root = tmp_path / "src"
    _clean(root / "t.parquet")
    _tag(root, tag)
    _refused(root, oracle_cfg)


def test_tag_nested_inside_a_directory_source_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    root = tmp_path / "src"
    _clean(root / "clean" / "t.parquet")
    _clean(root / "sub" / "deep" / "t.parquet")
    _tag(root / "sub" / "deep", tag)
    assert str(root / "sub" / "deep") in _refused(root, oracle_cfg)


def test_zero_byte_tag_is_refused_although_polars_reads_the_directory(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    root = tmp_path / "src"
    _clean(root / "t.parquet")
    _tag(root, tag, empty=True)
    assert pl.scan_parquet(root).collect().height == 3  # polars alone would ingest it
    _refused(root, oracle_cfg)


def test_symlinked_subdirectory_into_a_tagged_tree_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    store_day = tmp_path / "store" / "day"
    _clean(store_day / "o.parquet")
    _tag(store_day, tag)
    root = tmp_path / "src"
    _clean(root / "t.parquet")
    (root / "linked").symlink_to(tmp_path / "store", target_is_directory=True)
    _refused(root, oracle_cfg)


def test_file_symlink_pointing_into_a_tagged_directory_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    store_day = tmp_path / "store" / "day"
    target = _clean(store_day / "o.parquet")
    _tag(store_day, tag)
    link = tmp_path / "elsewhere" / "o.parquet"
    link.parent.mkdir()
    link.symlink_to(target)
    assert str(store_day) in _refused(link, oracle_cfg)


def test_symlink_pointing_below_a_tagged_directory_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    """Phase 3 amendment: the tag sits ABOVE the symlink target, inside nothing the walk visits."""
    store_day = tmp_path / "store" / "day"
    _tag(store_day, tag)
    _clean(store_day / "sub" / "o.parquet")
    root = tmp_path / "src"
    _clean(root / "t.parquet")
    (root / "linked").symlink_to(store_day / "sub", target_is_directory=True)
    assert str(store_day) in _refused(root, oracle_cfg)


def test_dangling_symlink_tag_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    day = tmp_path / "day"
    p = _clean(day / "t.parquet")
    (day / tag).symlink_to(tmp_path / "does-not-exist")
    _refused(p, oracle_cfg)


def test_symlink_cycle_terminates_and_still_finds_the_tag(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    root = tmp_path / "src"
    _clean(root / "t.parquet")
    (root / "loop").symlink_to(root, target_is_directory=True)
    _tag(root / "deep", tag)
    _refused(root, oracle_cfg)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs non-root POSIX perms")
def test_unreadable_subdirectory_fails_closed(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig
) -> None:
    root = tmp_path / "src"
    _clean(root / "t.parquet")
    locked = root / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        _refused(root, oracle_cfg)
    finally:
        locked.chmod(0o755)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs non-root POSIX perms")
def test_unsearchable_ancestor_fails_closed(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig
) -> None:
    """``os.path.lexists`` would report a tag here as absent (fail open); the check must refuse."""
    day = tmp_path / "day"
    _clean(day / "t.parquet")
    day.chmod(0o600)  # readable, not searchable: lstat of any child raises EACCES
    try:
        assert "could not be verified clean" in _refused(day / "t.parquet", oracle_cfg)
    finally:
        day.chmod(0o755)


def test_relative_source_is_checked_against_its_real_ancestors(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tag(tmp_path / "store", tag)
    work = tmp_path / "store" / "work"
    _clean(work / "data" / "t.parquet")
    monkeypatch.chdir(work)
    _refused(Path("data/t.parquet"), oracle_cfg)


def test_tilde_source_is_expanded_before_the_walk(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    _tag(home / "store", tag)
    _clean(home / "store" / "t.parquet")
    monkeypatch.setenv("HOME", str(home))
    _refused(Path("~/store/t.parquet"), oracle_cfg)


# ---------------------------------------------------------------------------
# Unsupported sources + fail-closed config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["*.parquet", "t?.parquet", "t[0].parquet"])
def test_glob_source_is_refused(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, pattern: str
) -> None:
    _clean(tmp_path / "t0.parquet")
    _refused(tmp_path / pattern, oracle_cfg)


def test_missing_oracle_config_is_refused_not_disabled(tmp_path: Path) -> None:
    p = _clean(tmp_path / "t.parquet")
    assert "[data.oracle]" in _refused(p, None)


# ---------------------------------------------------------------------------
# Through the data layer (no phantom implementations)
# ---------------------------------------------------------------------------


def test_load_parquet_refuses_before_scanning(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    root = tmp_path / "src"
    _clean(root / "t.parquet")
    _tag(root, tag)  # non-empty: polars alone raises InvalidOperationError here
    with pytest.raises(OracleQuarantineError):
        load_parquet(root, oracle=oracle_cfg)


def test_load_parquet_refuses_oracle_column(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, ns: str
) -> None:
    p = tmp_path / "t.parquet"
    pl.DataFrame({"x": [1], f"{ns}y": [1]}).write_parquet(p)
    with pytest.raises(OracleQuarantineError):
        load_parquet(p, oracle=oracle_cfg)


def test_compute_data_hash_and_data_hashes_refuse(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    day = tmp_path / "day"
    p = _clean(day / "t.parquet")
    _tag(day, tag)
    with pytest.raises(OracleQuarantineError):
        compute_data_hash(p, oracle=oracle_cfg)
    with pytest.raises(OracleQuarantineError):
        data_hashes(p, oracle=oracle_cfg)


def test_snapshot_refuses_and_writes_nothing(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig, tag: str
) -> None:
    day = tmp_path / "day"
    p = _clean(day / "t.parquet")
    _tag(day, tag)
    cas, manifests = tmp_path / "cas", tmp_path / "manifests"
    with pytest.raises(OracleQuarantineError):
        snapshot(name="n", source_path=p, cas_root=cas, manifests_root=manifests, oracle=oracle_cfg)
    assert not cas.exists()
    assert not manifests.exists()


# ---------------------------------------------------------------------------
# Clean-path regression — the regression that matters
# ---------------------------------------------------------------------------


def test_clean_source_loads_an_identical_frame(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig
) -> None:
    p = _clean(tmp_path / "t.parquet")
    assert materialize(load_parquet(p, oracle=oracle_cfg)).equals(pl.read_parquet(p))


def test_clean_hive_directory_loads_unchanged(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig
) -> None:
    root = tmp_path / "hive"
    _clean(root / "k=1" / "p.parquet")
    got = materialize(load_parquet(root, oracle=oracle_cfg))
    assert got.equals(pl.scan_parquet(root).collect())
    assert "k" in got.columns


def test_golden_data_hash_is_unchanged(oracle_cfg: OracleQuarantineConfig) -> None:
    pinned = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    got = compute_data_hash(GOLDEN_DIR / "synthetic.parquet", oracle=oracle_cfg)
    assert got["bytes_hash"] == pinned["data_bytes_hash"]
    assert got["logical_hash"] == pinned["data_logical_hash"]
    assert (
        data_hashes(GOLDEN_DIR / "synthetic.parquet", oracle=oracle_cfg)["data_hash"]
        == (pinned["data_hash"])
    )


def test_clean_snapshot_manifest_matches_pinned_hashes(
    tmp_path: Path, oracle_cfg: OracleQuarantineConfig
) -> None:
    pinned = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    src = tmp_path / "synthetic.parquet"
    src.write_bytes((GOLDEN_DIR / "synthetic.parquet").read_bytes())
    m = snapshot(
        name="golden",
        source_path=src,
        cas_root=tmp_path / "cas",
        manifests_root=tmp_path / "manifests",
        oracle=oracle_cfg,
    )
    assert m.bytes_hash == pinned["data_bytes_hash"]
    assert m.logical_hash == pinned["data_logical_hash"]
