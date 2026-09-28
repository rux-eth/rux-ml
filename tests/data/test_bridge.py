"""PR-047: the training-set data-hash bridge (program PR-024 amendment A13; C6 / C11).

rux-ml computes its own ``data_hash`` per training subtree in its own environment
and records it in a sidecar with the rux-ml commit SHA and the Polars version
(Polars guarantees its row hash only within one version, so the harness never
re-implements it). The equality test compares the harness manifest's file list +
sha256 with the files rux-ml's loader actually opens — read from Polars itself —
and with rux-ml's ``data_hash``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rux_ml._internal.git import git_sha
from rux_ml.data.bridge import BridgeError, check_bridge, loader_file_set, training_set_sidecar
from rux_ml.data.loaders import load_parquet, materialize
from rux_ml.runs.provenance import data_hashes


def _subtree(root: Path) -> Path:
    walk = root / "walk"
    for day in range(3):
        d = walk / f"day={day}"
        d.mkdir(parents=True)
        pl.DataFrame(
            {"stamp_ms": [day * 10 + i for i in range(4)], "y__walk_bp": [1.0] * 4}
        ).write_parquet(d / "part-0.parquet")
    return walk


def _files(source: Path) -> list[dict[str, object]]:
    return [
        {"path": str(p.relative_to(source)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(source.rglob("*.parquet"))
    ]


def _manifest(source: Path, sidecar: dict[str, object] | None = None) -> dict[str, Any]:
    """The harness set manifest's shape (program PR-024 ``write_manifest``): the subtree's
    files under ``subtrees.<t>`` (with the harness's extra keys) and rux-ml's listed
    sidecar under ``ruxml_sidecars.<t>`` — ``null`` until the harness lists it."""
    t = source.name
    files = [{**f, "day": "d", "rows": 4, "bytes": 1} for f in _files(source)]
    listed = None if sidecar is None else {"path": f"{t}.ruxml-sidecar.json", "sha256": "0" * 64,
                                           "data_hash": sidecar["data_hash"]}  # fmt: skip
    return {
        "set": "s",
        "subtrees": {t: {"subtree": f"{t}/", "files": files, "view": f"{t}.manifest.json"}},
        "ruxml_sidecars": {t: listed},
    }


def test_the_loader_file_set_is_what_polars_reads(tmp_path: Path, oracle_cfg: object) -> None:
    src = _subtree(tmp_path)
    got = loader_file_set(src)
    assert got == ["day=0/part-0.parquet", "day=1/part-0.parquet", "day=2/part-0.parquet"]
    # the same scan the loader runs: same rows
    assert materialize(load_parquet(src, oracle=oracle_cfg)).height == 12  # type: ignore[arg-type]


def test_the_sidecar_carries_the_hash_the_files_and_the_environment(
    tmp_path: Path, oracle_cfg: object
) -> None:
    src = _subtree(tmp_path)
    side = training_set_sidecar(src, oracle=oracle_cfg)  # type: ignore[arg-type]
    assert side["data_hash"] == data_hashes(src, oracle=oracle_cfg)["data_hash"]  # type: ignore[arg-type]
    assert side["polars_version"] == pl.__version__
    assert side["rux_ml_git_sha"] == git_sha()
    assert side["row_count"] == 12
    assert side["files"] == _files(src)
    assert side["hashed_files"] == side["opened_files"] == [f["path"] for f in side["files"]]  # type: ignore[index]


def test_the_bridge_holds_on_an_equal_manifest(tmp_path: Path, oracle_cfg: object) -> None:
    src = _subtree(tmp_path)
    side = training_set_sidecar(src, oracle=oracle_cfg)  # type: ignore[arg-type]
    result = check_bridge(side, _manifest(src, side), subtree="walk")
    assert result == {"subtree": "walk", "files": 3, "data_hash": side["data_hash"], "equal": True}


@pytest.mark.parametrize("listed", ["null", "no_entry", "no_ruxml_sidecars"])
def test_the_bridge_fails_closed_without_the_harness_listed_data_hash(
    tmp_path: Path, oracle_cfg: object, listed: str
) -> None:
    """PR-052: the harness lists rux-ml's data_hash at ``ruxml_sidecars.<t>.data_hash``;
    while it is absent the test refuses — the file list alone never passes."""
    src = _subtree(tmp_path)
    side = training_set_sidecar(src, oracle=oracle_cfg)  # type: ignore[arg-type]
    man = _manifest(src)  # ruxml_sidecars.walk = null: the sidecar not yet listed
    if listed == "no_entry":
        man["ruxml_sidecars"] = {}
    elif listed == "no_ruxml_sidecars":
        del man["ruxml_sidecars"]
    with pytest.raises(BridgeError, match=r"no rux-ml data_hash for walk/"):
        check_bridge(side, man, subtree="walk")


def test_the_pr047_assumed_shape_is_not_read(tmp_path: Path, oracle_cfg: object) -> None:
    """PR-052: nothing real writes a top-level ``files`` + ``ruxml_data_hash`` (the harness
    asserts its view has no ``ruxml_data_hash``); that shape — which passed on the file list
    — is refused."""
    src = _subtree(tmp_path)
    side = training_set_sidecar(src, oracle=oracle_cfg)  # type: ignore[arg-type]
    assumed = {"files": _files(src), "ruxml_data_hash": side["data_hash"]}
    with pytest.raises(BridgeError, match=r"subtrees\.walk\.files"):
        check_bridge(side, assumed, subtree="walk")


@pytest.mark.parametrize("defect", ["missing", "extra", "sha", "data_hash"])
def test_the_bridge_refuses_any_difference(tmp_path: Path, oracle_cfg: object, defect: str) -> None:
    src = _subtree(tmp_path)
    side = training_set_sidecar(src, oracle=oracle_cfg)  # type: ignore[arg-type]
    man = _manifest(src, side)
    files = man["subtrees"]["walk"]["files"]
    assert isinstance(files, list)
    if defect == "missing":
        files.pop()
    elif defect == "extra":
        files.append({"path": "day=9/part-0.parquet", "sha256": "0" * 64})
    elif defect == "sha":
        files[0] = {**files[0], "sha256": "0" * 64}
    else:
        man["ruxml_sidecars"]["walk"]["data_hash"] = "x|y"
    with pytest.raises(BridgeError):
        check_bridge(side, man, subtree="walk")


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_a_file_the_loader_opens_but_the_hash_misses_is_refused(
    tmp_path: Path, oracle_cfg: object
) -> None:
    """PR-040's successor (ii): Polars follows a symlinked subdirectory that
    ``rglob`` (the hash's file list) does not — the sidecar must not claim a hash
    that does not cover the loaded files. (Plain directories: Polars refuses a tree
    mixing hive ``day=`` directories with a non-hive one.)"""
    src = tmp_path / "walk"
    for day in range(2):
        (src / f"d{day}").mkdir(parents=True)
        pl.DataFrame({"stamp_ms": [day], "y__walk_bp": [1.0]}).write_parquet(
            src / f"d{day}" / "p.parquet"
        )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pl.DataFrame({"stamp_ms": [99], "y__walk_bp": [2.0]}).write_parquet(elsewhere / "extra.parquet")
    (src / "linked").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(BridgeError, match=r"linked/extra\.parquet"):
        training_set_sidecar(src, oracle=oracle_cfg)  # type: ignore[arg-type]


def test_the_manifest_json_round_trips(tmp_path: Path, oracle_cfg: object) -> None:
    src = _subtree(tmp_path)
    side = training_set_sidecar(src, oracle=oracle_cfg)  # type: ignore[arg-type]
    assert json.loads(json.dumps(side)) == side
