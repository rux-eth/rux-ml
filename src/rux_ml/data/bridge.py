"""The training-set data-hash bridge (PR-047; program PR-024 amendment A13, C6 / C11).

The harness materializes the training set (program PR-024: subtrees ``fill/`` and
``walk/``) with a manifest listing every file and its sha256. rux-ml's ``data_hash``
includes a Polars row hash that Polars guarantees only within one version, so the
harness never re-implements it: rux-ml computes it here, in its own environment,
and records it in a **sidecar** with the rux-ml commit SHA and the Polars version.

The **equality test** (:func:`check_bridge`, PR-052) runs against the harness's **set
manifest** (``<set>/manifest.json``): its ``subtrees.<subtree>.files`` (path relative to
the subtree + sha256) must equal the files rux-ml's loader opens — read from Polars
itself, via ``include_file_paths`` on the loader's own scan — with their sha256, and its
``ruxml_sidecars.<subtree>.data_hash`` (the harness's copy of rux-ml's ``data_hash``,
listed once rux-ml's sidecar exists) must equal the sidecar's. It fails closed: while
the harness has not listed the sidecar, the test refuses rather than passing on the
file list alone. The sidecar itself refuses a source whose loaded files differ from the
files the hash covers (PR-040's named successor (ii)): a hash that does not cover what
was loaded is not a hash of the training set.

**The harness manifest id** (PR-051, :func:`harness_manifest_ref`): the materializer
writes the set's ``manifest.json`` at the set root and one view per subtree beside it,
``<set>/<subtree>.manifest.json``, citing the set manifest by name and sha256. That
sha256 is the id a promoted model records; it is verified against the file.

The harness shape (program PR-024, rumpy-harness ``training/materialize.py``
``write_manifest``; checked against the real set 2026-09-27): ``{"set", "subtrees":
{<t>: {"files": [{"path", "sha256", ...}], ...}}, "ruxml_sidecars": {<t>: null |
{"path", "sha256", "data_hash", "rux_ml_git_sha", "polars_version"}}, ...}``. Nothing
writes a top-level ``files`` / ``ruxml_data_hash`` (PR-047's assumed shape, removed).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from rux_ml._internal.git import git_sha
from rux_ml.data.loaders import iter_parquet_files
from rux_ml.data.versioning import compute_data_hash

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rux_ml.config.data import OracleQuarantineConfig

# A temporary column naming each row's source file; refused if the set has one.
_FILE_COL = "__ruxml_loader_source_file__"
_CHUNK = 1 << 20
# The harness's view of one subtree, beside it at the set root (program PR-024's
# materializer: ``subtrees[t]["view"] = f"{t}.manifest.json"``) — its file contract.
_VIEW_SUFFIX = ".manifest.json"


class BridgeError(ValueError):
    """The set, its hash and the harness manifest do not agree (a ``ValueError``: exit 2)."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"data bridge: {detail} — refused")


class HarnessManifestError(BridgeError):
    """The harness manifest a model would record does not describe its training set."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"harness manifest: {detail}")


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


def listed_data_hash(manifest: Mapping[str, Any], subtree: str) -> str | None:
    """The harness set manifest's copy of rux-ml's ``data_hash`` for ``subtree``
    (``ruxml_sidecars.<subtree>.data_hash``), or ``None`` while it lists no sidecar."""
    sidecars: object = manifest.get("ruxml_sidecars")
    listed: object = (
        cast("dict[str, object]", sidecars).get(subtree) if isinstance(sidecars, dict) else None
    )
    theirs: object = (
        cast("dict[str, object]", listed).get("data_hash") if isinstance(listed, dict) else None
    )
    return theirs if isinstance(theirs, str) else None


def check_bridge(
    sidecar: Mapping[str, Any], manifest: Mapping[str, Any], *, subtree: str
) -> dict[str, Any]:
    """The equality test against the harness **set manifest** (``<set>/manifest.json``):
    ``subtrees.<subtree>.files`` (path + sha256) vs the files rux-ml's loader opened with
    their sha256, and ``ruxml_sidecars.<subtree>.data_hash`` vs rux-ml's ``data_hash``.
    Any difference raises :class:`BridgeError`, and so does an absent ``data_hash``
    (the harness has not listed the sidecar yet): never a pass on the file list alone."""
    subtrees: object = manifest.get("subtrees")
    entry: object = (
        cast("dict[str, object]", subtrees).get(subtree) if isinstance(subtrees, dict) else None
    )
    raw: object = cast("dict[str, object]", entry).get("files") if isinstance(entry, dict) else None
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
        msg = (
            f"the manifest has no subtrees.{subtree}.files list of {{path, sha256}} "
            f"(pass the harness set's manifest.json)"
        )
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
    theirs_hash = listed_data_hash(manifest, subtree)
    if theirs_hash is None:
        msg = (
            f"the harness manifest lists no rux-ml data_hash for {subtree}/ "
            f"(ruxml_sidecars.{subtree}): write the sidecar, let the harness list it, then check"
        )
        raise BridgeError(msg)
    if theirs_hash != sidecar["data_hash"]:
        msg = (
            f"harness ruxml_sidecars.{subtree}.data_hash {theirs_hash} != rux-ml data_hash "
            f"{sidecar['data_hash']}"
        )
        raise BridgeError(msg)
    return {"subtree": subtree, "files": len(ours), "data_hash": theirs_hash, "equal": True}


def harness_view_path(source: Path) -> Path:
    """Where the harness's view of the subtree ``source`` is: ``<set>/<subtree>.manifest.json``."""
    return source.parent / f"{source.name}{_VIEW_SUFFIX}"


def harness_manifest_ref(source: Path) -> dict[str, Any] | None:
    """The harness manifest of the set ``source`` is a subtree of, or ``None`` if the
    subtree has no harness view (not a harness set).

    Returns ``manifest_id`` (the sha256 of the set manifest the view cites, verified
    against the file), ``set_name``, ``subtree``, ``view_sha256`` and ``ruxml_data_hash``
    (the set manifest's copy of rux-ml's ``data_hash`` for the subtree, ``None`` before
    rux-ml's sidecar is listed). Refuses (:class:`HarnessManifestError`) a view that
    names another subtree or a missing set manifest, or cites a set manifest that has
    been rewritten since the view was written.
    """
    view_path = harness_view_path(source)
    if not view_path.is_file():
        return None
    view = cast("dict[str, Any]", json.loads(view_path.read_text()))
    name, cited = view.get("set_manifest"), view.get("set_manifest_sha256")
    if view.get("subtree") != source.name:
        msg = f"{view_path} is the view of subtree {view.get('subtree')!r}, not {source.name!r}"
        raise HarnessManifestError(msg)
    if not isinstance(name, str) or Path(name).name != name or not isinstance(cited, str):
        msg = f"{view_path} does not name the set manifest (set_manifest, set_manifest_sha256)"
        raise HarnessManifestError(msg)
    set_path = source.parent / name
    if not set_path.is_file():
        msg = f"{view_path} names {name}, which is not at {set_path}"
        raise HarnessManifestError(msg)
    actual = _sha256(set_path)
    if actual != cited:
        msg = (
            f"{view_path} cites {name} sha256 {cited}, the file is {actual}: the set manifest "
            f"was rewritten after the view"
        )
        raise HarnessManifestError(msg)
    body = cast("dict[str, Any]", json.loads(set_path.read_text()))
    set_name = body.get("set")
    if not isinstance(set_name, str):
        msg = f"{set_path} has no set name"
        raise HarnessManifestError(msg)
    return {
        "manifest_id": actual,
        "set_name": set_name,
        "subtree": source.name,
        "view_sha256": _sha256(view_path),
        "ruxml_data_hash": listed_data_hash(body, source.name),
    }
