"""Content-addressed dataset versioning per D9 + D14.

Composite ``data_hash`` = ``bytes_hash`` (layout-sensitive) + ``logical_hash``
(layout-invariant, computed one file at a time since PR-053), with the schema and row count
recorded alongside. Snapshots
hardlink into a CAS (``data/cas/...``) with a JSON manifest under
``data/manifests/<name>/``. Hardlinks fall back to ``shutil.copy2`` if source
and destination live on different filesystems (``OSError(errno.EXDEV)``).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from rux_ml._internal.hashing import canonical_json, xxh3_64_file
from rux_ml.data.loaders import iter_parquet_files
from rux_ml.data.quarantine import check_oracle_quarantine

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rux_ml.config.data import OracleQuarantineConfig


class Manifest(BaseModel):
    """Per-version dataset manifest written to ``data/manifests/<name>/<version_id>.json``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version_id: str
    bytes_hash: str
    logical_hash: str
    row_count: int
    schema_: dict[str, str]  # field name -> dtype string; aliased to avoid shadowing
    source_path: str
    cas_files: list[str]
    created_at: str  # ISO 8601 UTC


# ---------- Hashing ----------


def _bytes_hash(files: Iterable[Path]) -> str:
    """SHA-256 of the canonical-JSON of sorted per-file xxh3_64 digests."""
    digests = [xxh3_64_file(f) for f in files]
    return hashlib.sha256(canonical_json(sorted(digests)).encode()).hexdigest()


def _footer_rows(path: Path) -> int:
    """One file's row count, from its footer (no row data is read)."""
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _logical_hash(files: list[Path]) -> tuple[str, int, dict[str, str]]:
    """Layout-invariant row hash + row count + schema map, computed one file at a time (PR-053).

    Each row is hashed as a Polars struct over the sorted column names (``hash(seed=0)``). The
    u64 row hashes of every file are sorted together and SHA-256'd as raw bytes. A row's hash
    depends only on that row, and the sort removes order, so the digest does not depend on how
    rows are split into files. It is bit-identical to ``_logical_hash_whole_source``, the
    computation it replaces (tests/data/test_logical_hash_per_file.py).

    Memory: Polars is handed one file at a time, and only that file's hash column is collected.
    The whole source is held only as one u64 per row, in one preallocated array that is sorted in
    place. On program PR-027's 156-day fill view (95.4M rows) the whole-source computation peaked
    at 18.0 GiB and the per-file one at 7.3 GiB, with the same digest. The array is freed when
    this function returns, before a caller loads the data for its fit.

    Each file is scanned with the first file's schema (``schema=``). One scan over every file
    applies that same check to every later file, so the per-file path refuses what it refused
    (a missing column: ``ColumnNotFoundError``; an extra column or a dtype difference:
    ``SchemaError``, now naming the file). It also accepts what it accepted: a different column
    order, a different struct field order, a tz-naive datetime under a tz-aware first file.
    """
    if not files:
        msg = "no parquet files found"
        raise FileNotFoundError(msg)
    reference = pl.scan_parquet(files[0]).collect_schema()
    cols = sorted(reference.names())
    schema_map = {name: str(reference[name]) for name in cols}

    counts = [_footer_rows(f) for f in files]
    row_h = np.empty(sum(counts), dtype=np.uint64)
    at = 0
    for f, rows in zip(files, counts, strict=True):
        try:
            part = (
                pl.scan_parquet(f, schema=reference)
                .select(pl.struct(cols).hash(seed=0).alias("_h"))
                .collect(engine="streaming")["_h"]
            )
        except (pl.exceptions.SchemaError, pl.exceptions.ColumnNotFoundError) as exc:
            msg = f"{f}: its schema differs from the first file's ({files[0]}): {exc}"
            raise type(exc)(msg) from exc
        if part.len() != rows:
            msg = f"{f}: {part.len()} rows hashed but its footer declared {rows}; did it change?"
            raise RuntimeError(msg)
        row_h[at : at + rows] = part.to_numpy()
        at += rows
        del part
    row_h.sort()
    digest = hashlib.sha256(memoryview(row_h).cast("B")).hexdigest()
    row_count = len(row_h)
    del row_h
    return digest, row_count, schema_map


def _logical_hash_whole_source(files: list[Path]) -> tuple[str, int, dict[str, str]]:  # pyright: ignore[reportUnusedFunction]  -- the reference: tests + scripts only
    """The pre-PR-053 ``_logical_hash``, kept verbatim as the reference. Not on any runtime path.

    Only the equality tests and ``scripts/check_logical_hash_per_file.py`` call it. That script is
    the real-set check of program PR-027 A2. This path loads the whole source into one scan and
    sorts it: 18.0 GiB on the 156-day fill view.
    """
    if not files:
        msg = "no parquet files found"
        raise FileNotFoundError(msg)
    # Pass the explicit file list so we don't accidentally scan a parent
    # directory that may hold other unrelated parquet files. Polars accepts
    # both a single Path and a list of Paths.
    scan_target: Path | list[Path] = files[0] if len(files) == 1 else files
    lf = pl.scan_parquet(scan_target)
    schema_obj = lf.collect_schema()
    cols = sorted(schema_obj.names())
    schema_map = {name: str(schema_obj[name]) for name in cols}

    row_h = (
        lf.select(pl.struct(cols).hash(seed=0).alias("_h")).sort("_h").collect(engine="streaming")
    )
    arr = row_h["_h"].to_numpy()
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    row_count = int(row_h.height)
    return digest, row_count, schema_map


def compute_data_hash(path: Path, *, oracle: OracleQuarantineConfig | None) -> dict[str, Any]:
    """Compute the composite data hash + schema metadata for a Parquet path.

    Returns a dict with keys: ``bytes_hash``, ``logical_hash``, ``row_count``,
    ``schema``. Both hashes are SHA-256 hex; ``schema`` is name -> dtype-string.

    Runs the oracle quarantine (PR-040) first; it only raises, so hash inputs
    are unchanged for any source it lets through.
    """
    return compute_data_hash_and_files(path, oracle=oracle)[0]


def compute_data_hash_and_files(
    path: Path, *, oracle: OracleQuarantineConfig | None
) -> tuple[dict[str, Any], list[Path]]:
    """:func:`compute_data_hash` plus the file list the hash covered, in hash order (PR-059;
    program PR-028a Q6): the list a reader compares its own against, never a second
    enumeration of the source."""
    check_oracle_quarantine(path, oracle)
    files = iter_parquet_files(path)
    bytes_h = _bytes_hash(files)
    logical_h, row_count, schema_map = _logical_hash(files)
    composite = {
        "bytes_hash": bytes_h,
        "logical_hash": logical_h,
        "row_count": row_count,
        "schema": schema_map,
    }
    return composite, list(files)


# ---------- CAS + manifest ----------


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` -> ``dst``; fall back to ``shutil.copy2`` on EXDEV.

    Per the PR-004 Phase 1 documentation note: hardlinks require src and dst on
    the same filesystem. On a cross-device error we degrade to a content-
    identical copy and emit a warning to stderr.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        warnings.warn(
            f"cross-device link from {src} to {dst}; falling back to copy",
            stacklevel=2,
        )
        shutil.copy2(src, dst)


def _cas_dest(cas_root: Path, file_bytes_hash: str, original_name: str) -> Path:
    """Two-level fan-out in CAS: ``cas/<aa>/<aaaa...>__<original_name>``."""
    return cas_root / file_bytes_hash[:2] / f"{file_bytes_hash}__{original_name}"


def snapshot(
    name: str,
    source_path: Path,
    *,
    cas_root: Path,
    manifests_root: Path,
    oracle: OracleQuarantineConfig | None,
) -> Manifest:
    """Snapshot ``source_path`` into the CAS and write a manifest under ``name``.

    The ``version_id`` is the prefix of the composite hash ``sha256(bytes_hash +
    "|" + logical_hash)[:16]`` — short enough for filenames, stable across
    re-snapshots of identical content.
    """
    files = iter_parquet_files(source_path)
    composite = compute_data_hash(source_path, oracle=oracle)
    version_id = hashlib.sha256(
        f"{composite['bytes_hash']}|{composite['logical_hash']}".encode()
    ).hexdigest()[:16]

    cas_files: list[str] = []
    for f in files:
        per_file_hash = xxh3_64_file(f)
        dest = _cas_dest(cas_root, per_file_hash, f.name)
        _link_or_copy(f, dest)
        cas_files.append(str(dest))

    manifest = Manifest(
        name=name,
        version_id=version_id,
        bytes_hash=composite["bytes_hash"],
        logical_hash=composite["logical_hash"],
        row_count=composite["row_count"],
        schema_=composite["schema"],
        source_path=str(source_path),
        cas_files=cas_files,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    write_manifest(manifest, manifests_root)
    return manifest


def write_manifest(manifest: Manifest, manifests_root: Path) -> Path:
    """Atomically write a manifest JSON to ``manifests_root/<name>/<version_id>.json``."""
    out_dir = manifests_root / manifest.name
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{manifest.version_id}.json"
    tmp = final.with_suffix(".json.tmp")
    payload = manifest.model_dump(mode="json")
    # Surface the field as "schema" in the JSON even though Python uses schema_.
    payload["schema"] = payload.pop("schema_")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(final)
    return final


def read_manifest(path: Path) -> Manifest:
    """Load a manifest JSON, validating against the Pydantic schema."""
    raw = json.loads(Path(path).read_text())
    # Translate the JSON "schema" key back to the Python "schema_" attr.
    if "schema" in raw:
        raw["schema_"] = raw.pop("schema")
    return Manifest.model_validate(raw)


def list_manifests(manifests_root: Path, *, name: str | None = None) -> list[Manifest]:
    """Return all manifests under ``manifests_root``, optionally filtered by name."""
    root = Path(manifests_root)
    if not root.exists():
        return []
    targets: Iterable[Path]
    if name is not None:
        targets = sorted((root / name).glob("*.json")) if (root / name).is_dir() else []
    else:
        targets = sorted(root.rglob("*.json"))
    return [read_manifest(p) for p in targets]
