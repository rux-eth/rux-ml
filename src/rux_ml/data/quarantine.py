"""Oracle quarantine at ingest (PR-040; program ACCEPTANCE C13).

The rux-capital harness builds perfect-foresight *oracle* labels in a separate,
tagged store and refuses to serve them. This module is the independent second
refusal: no training set may be built from a source that carries the oracle tag
file or an oracle-namespace column. The rule mirrors the harness's
``production_load`` but is deliberately stricter (path normalization, symlink
following, hive keys, nested fields, case-insensitive matching).

The check is raise-only: it never changes the scan target, the scan options or
any hash input, so clean-path frames and hashes are unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from rux_ml.config.data import OracleQuarantineConfig

# Polars' own glob test (crates/polars-io/src/path_utils/mod.rs:128-133 @ py-1.40.1).
_GLOB_CHARS = ("*", "?", "[")


class OracleQuarantineError(ValueError):
    """A training-set source was refused as oracle-derived (or could not be verified clean).

    A ``ValueError`` so the CLI's existing ``ValueError`` -> ``typer.BadParameter``
    mapping reports it with exit code 2.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"oracle quarantine: {detail} — refused")


def check_oracle_quarantine(path: Path, oracle: OracleQuarantineConfig | None) -> None:
    """Raise :class:`OracleQuarantineError` if ``path`` is, or may be, oracle-derived.

    Refused when:

    1. ``oracle`` is ``None`` — a missing ``[data.oracle]`` table refuses rather
       than disabling the check.
    2. ``path`` is a glob.
    3. The tag file is present (symlinks not followed) in any ancestor of the
       path as given or of its resolved path; in any directory inside a
       directory source (the walk follows symlinks, cycle-guarded); or in any
       ancestor of the resolved target of a symlink met during the walk.
    4. Any column name, or nested struct/list/array field name, starts
       case-insensitively with the namespace — checked on every file's footer
       and on the source's own schema (which carries hive partition keys).

    Tag checks run before any polars call: polars itself raises on a non-empty
    tag file inside a directory source. ``RuntimeError`` / ``OSError`` raised
    while verifying are re-raised as :class:`OracleQuarantineError` (fail closed).
    """
    if oracle is None:
        msg = (
            f"{path}: no [data.oracle] table in the config (configs/base.toml sets it); "
            "ingest refuses rather than running unchecked"
        )
        raise OracleQuarantineError(msg)
    raw = str(path)
    if any(ch in raw for ch in _GLOB_CHARS):
        msg = f"{raw}: glob sources are not supported"
        raise OracleQuarantineError(msg)
    try:
        source = Path(raw).expanduser()
        files = _check_tags(source, oracle.tag_file)
    except (RuntimeError, OSError) as exc:
        msg = f"{raw}: could not be verified clean ({exc!r})"
        raise OracleQuarantineError(msg) from exc
    _check_columns(source, files, oracle.namespace)


def _tagged_ancestor(path: Path, tag_file: str) -> Path | None:
    for directory in (path, *path.parents):
        if (directory / tag_file).exists(follow_symlinks=False):
            return directory
    return None


def _check_ancestors(path: Path, tag_file: str, label: str) -> None:
    for chain in (path.absolute(), path.resolve()):
        tagged = _tagged_ancestor(chain, tag_file)
        if tagged is not None:
            msg = f"{label}: under the oracle-tagged directory {tagged}"
            raise OracleQuarantineError(msg)


def _raise(err: OSError) -> None:
    raise err


def _check_tags(source: Path, tag_file: str) -> list[Path]:
    """Refuse on any tag reachable from ``source``; return the files polars would read.

    Polars reads every non-directory entry larger than 0 bytes under a directory
    source, following symlinks (path_utils/mod.rs:477-479 @ py-1.40.1). The walk
    visits at least that set, with a ``(st_dev, st_ino)`` guard against symlink
    cycles (the pattern at setuptools ``_distutils/filelist.py``, bpo-44497).
    """
    _check_ancestors(source, tag_file, str(source))
    if not source.is_dir():
        return [source]
    seen: set[tuple[int, int]] = set()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(source, followlinks=True, onerror=_raise):
        st = os.stat(dirpath)
        key = (st.st_dev, st.st_ino)
        if key in seen:
            del dirnames[:]
            continue
        seen.add(key)
        directory = Path(dirpath)
        if (directory / tag_file).exists(follow_symlinks=False):
            msg = f"{source}: contains the oracle-tagged directory {directory}"
            raise OracleQuarantineError(msg)
        for name in (*dirnames, *filenames):
            entry = directory / name
            if entry.is_symlink():
                _check_ancestors(entry.resolve(), tag_file, f"{source}: symlink {entry}")
        files.extend(
            directory / name for name in filenames if (directory / name).stat().st_size > 0
        )
    return sorted(files)


def _namespace_hits(schema: pl.Schema, namespace: str) -> list[str]:
    folded = namespace.casefold()
    hits: list[str] = []

    # `object`: `List.inner` is typed PolarsDataType (may be a class); isinstance narrows.
    def walk(dtype: object, prefix: str) -> None:
        if isinstance(dtype, pl.Struct):
            for field in dtype.fields:
                name = f"{prefix}.{field.name}"
                if field.name.casefold().startswith(folded):
                    hits.append(name)
                walk(field.dtype, name)
        elif isinstance(dtype, (pl.List, pl.Array)):
            walk(dtype.inner, f"{prefix}[]")

    for name, dtype in schema.items():
        if name.casefold().startswith(folded):
            hits.append(name)
        walk(dtype, name)
    return hits


def _check_columns(source: Path, files: list[Path], namespace: str) -> None:
    hits = [
        f"{f}: {name}"
        for f in files
        for name in _namespace_hits(pl.scan_parquet(f, glob=False).collect_schema(), namespace)
    ]
    # The source-level schema adds hive partition keys (directory sources only).
    hits += [
        f"{source}: {name}"
        for name in _namespace_hits(pl.scan_parquet(source).collect_schema(), namespace)
        if not any(h.endswith(f": {name}") for h in hits)
    ]
    if hits:
        msg = f"oracle-namespace columns {hits}"
        raise OracleQuarantineError(msg)
