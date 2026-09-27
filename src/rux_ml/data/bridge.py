"""The training-set data-hash bridge (PR-047; program PR-024 amendment A13, C6 / C11).

The harness materializes the training set (program PR-024: subtrees ``fill/`` and
``walk/``) with a manifest listing every file and its sha256. rux-ml's ``data_hash``
includes a Polars row hash that Polars guarantees only within one version, so the
harness never re-implements it: rux-ml computes it here, in its own environment,
and records it in a **sidecar** with the rux-ml commit SHA and the Polars version.

The **equality test** (:func:`check_bridge`): the harness manifest's file list +
sha256 must equal the files rux-ml's loader opens — read from Polars itself, via
``include_file_paths`` on the loader's own scan — with their sha256, and the
manifest's copy of rux-ml's ``data_hash`` (when present) must equal the sidecar's.
The sidecar itself refuses a source whose loaded files differ from the files the
hash covers (PR-040's named successor (ii)): a hash that does not cover what was
loaded is not a hash of the training set.

Assumed harness manifest shape (program PR-024 not landed — flagged):
``{"files": [{"path": <relative to the subtree>, "sha256": <hex>}, ...],
"ruxml_data_hash": <optional>}``.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from rux_ml._internal.git import git_sha
from rux_ml.data.loaders import iter_parquet_files
from rux_ml.data.versioning import compute_data_hash

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from rux_ml.config.data import OracleQuarantineConfig

# A temporary column naming each row's source file; refused if the set has one.
_FILE_COL = "__ruxml_loader_source_file__"
_CHUNK = 1 << 20


class BridgeError(ValueError):
    """The set, its hash and the harness manifest do not agree (a ``ValueError``: exit 2)."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"data bridge: {detail} — refused")


def _root(source: Path) -> Path:
    return source if source.is_dir() else source.parent


def _rel(path: str | Path, root: Path) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root))


def loader_file_set(source: Path) -> list[str]:
    """The files the loader's scan reads rows from, relative to the subtree, sorted.

    The same ``pl.scan_parquet(source)`` as :func:`rux_ml.data.loaders.load_parquet`
    (default options), with ``include_file_paths``: Polars itself names the files.
    A zero-row file contributes no row and is therefore not listed.
    """
    lf = pl.scan_parquet(source, include_file_paths=_FILE_COL)
    if _FILE_COL in pl.scan_parquet(source).collect_schema().names():
        msg = f"the set already has a column {_FILE_COL!r}"
        raise BridgeError(msg)
    paths = lf.select(pl.col(_FILE_COL).unique()).collect(engine="streaming")[_FILE_COL]
    return sorted(_rel(p, _root(source)) for p in paths.to_list())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def training_set_sidecar(source: Path, *, oracle: OracleQuarantineConfig | None) -> dict[str, Any]:
    """rux-ml's record of one training subtree: its ``data_hash`` (composed exactly as
    ``runs.provenance.data_hashes``), every file with its sha256, the file lists the
    hash covers and the loader opens (refused unless equal), the rux-ml commit SHA
    and the Polars version. Runs the oracle quarantine first (via the hash)."""
    composite = compute_data_hash(source, oracle=oracle)
    root = _root(source)
    hashed_paths = iter_parquet_files(source)
    hashed = sorted(_rel(p, root) for p in hashed_paths)
    opened = loader_file_set(source)
    if hashed != opened:
        only_opened = sorted(set(opened) - set(hashed))
        only_hashed = sorted(set(hashed) - set(opened))
        msg = (
            f"the loader opens {only_opened} that the hash does not cover, and the hash "
            f"covers {only_hashed} that the loader does not open"
        )
        raise BridgeError(msg)
    by_rel = {_rel(p, root): p for p in hashed_paths}
    return {
        "source_path": str(source),
        "data_hash": f"{composite['bytes_hash']}|{composite['logical_hash']}",
        "data_bytes_hash": composite["bytes_hash"],
        "data_logical_hash": composite["logical_hash"],
        "row_count": composite["row_count"],
        "files": [{"path": rel, "sha256": _sha256(by_rel[rel])} for rel in hashed],
        "hashed_files": hashed,
        "opened_files": opened,
        "rux_ml_git_sha": git_sha(),
        "polars_version": pl.__version__,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def check_bridge(sidecar: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The equality test: the harness manifest's files + sha256 vs the files rux-ml's
    loader opened with their sha256, and the manifest's ``ruxml_data_hash`` (if any)
    vs rux-ml's ``data_hash``. Any difference raises :class:`BridgeError`."""
    raw: object = manifest.get("files")
    entries: list[dict[str, Any]] = (
        [cast("dict[str, Any]", f) for f in cast("list[object]", raw) if isinstance(f, dict)]
        if isinstance(raw, list)
        else []
    )
    if (
        not isinstance(raw, list)
        or len(entries) != len(cast("list[object]", raw))
        or not all({"path", "sha256"} <= set(f) for f in entries)
    ):
        msg = "the manifest has no files list of {path, sha256}"
        raise BridgeError(msg)
    theirs = {str(f["path"]): str(f["sha256"]) for f in entries}
    ours = {str(f["path"]): str(f["sha256"]) for f in sidecar["files"]}
    missing = sorted(set(ours) - set(theirs))
    extra = sorted(set(theirs) - set(ours))
    differ = sorted(p for p in set(ours) & set(theirs) if ours[p] != theirs[p])
    if missing or extra or differ:
        msg = (
            f"manifest vs loaded files: not in the manifest {missing}, not loaded {extra}, "
            f"sha256 differs {differ}"
        )
        raise BridgeError(msg)
    theirs_hash = manifest.get("ruxml_data_hash")
    if theirs_hash is not None and theirs_hash != sidecar["data_hash"]:
        msg = f"manifest ruxml_data_hash {theirs_hash} != rux-ml data_hash {sidecar['data_hash']}"
        raise BridgeError(msg)
    return {"files": len(ours), "data_hash_checked": theirs_hash is not None, "equal": True}
