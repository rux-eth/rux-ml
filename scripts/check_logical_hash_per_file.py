"""PR-053 / program PR-027 A2: on a real set, the per-file logical hash equals the whole-source one.

DEFERRED. Written 2026-09-28 and not yet run on the real set. It runs on the desktop once
PR-024's final set exists (a heavy job, so it takes ``~/rumpy-heavy.lock``). One command, from a
rux-ml checkout at PR-053 or later:

    mkdir ~/rumpy-heavy.lock && \
    echo "pr053-hash-check $(date -u +%FT%TZ)" > ~/rumpy-heavy.lock/owner && \
    systemd-run --user --scope -p MemoryHigh=20G -p MemoryMax=20G -p MemorySwapMax=0 \
      uv run python scripts/check_logical_hash_per_file.py \
        ~/projects/rux-capital/m9-training-v03/<label SHA>-<feature SHA> \
        --record ~/pr053-hash-check.json; \
    rc=$?; rm -rf ~/rumpy-heavy.lock; exit $rc

``SET`` is the set root, which holds ``manifest.json`` and one directory per subtree. The
subtrees are the manifest's ``subtrees`` keys, or those given with ``--subtree``. Each subtree's
logical hash is computed twice, and each computation runs in its own child process, so each has
its own peak RSS:

1. ``per_file``: ``versioning._logical_hash``, the path ``train``, ``tune``,
   ``registry promote`` and ``data bridge`` take since PR-053.
2. ``whole_source``: ``versioning._logical_hash_whole_source``, the pre-PR-053 computation kept
   as the reference.

It checks that digest, row count and schema are equal. When the set manifest lists rux-ml's
``data_hash`` for a subtree (``ruxml_sidecars.<t>.data_hash``, which the pre-PR-053 code wrote), it
also checks that the per-file composite ``bytes_hash|logical_hash`` equals it, so no recorded
value changes. Exit 0 when everything is equal, 1 when anything differs or a child failed (for
example, was killed by the scope), 2 on a usage error. The record (``--record``, and always
stdout) holds every value and each child's wall, peak RSS and the RSS left after the hash
returned.

EXPECTED (program PR-027 ``r2b/records/q3_hash_stream.json``, the 156-day fill view, 95,361,520
rows, desktop, Polars 1.40.1):

- per-file: about 48 s wall, cgroup peak 7.3 GiB.
- whole-source: about 10 s wall, 18.0 GiB (about 200 B per row).
- walk: about 1/5 of fill's rows, so about a fifth of each.
- Fill + walk: about 1.5 min in all, with the interpreter start-ups.

The whole-source child's peak grows with the rows. The script prints ``rows x 200 B`` for each
subtree before that child starts. A set much longer than 156 days can pass 20 GiB. The scope
then kills that child, which is recorded as a failure, and the host is untouched. Re-run with
``--skip-whole-source`` to compare the per-file digest with the manifest's recorded
``data_hash`` alone.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from rux_ml.data import versioning
from rux_ml.data.loaders import iter_parquet_files

WHOLE_SOURCE_BYTES_PER_ROW = 200  # measured: 18.0 GiB / 95.36M rows (program PR-027 q3_hash_stream)


def _peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (
        peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    )  # bytes on macOS, KiB on Linux


def _rss_now_mib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**10
    except OSError:
        return None
    return None


def _child(kind: str, source: Path) -> None:
    """Compute one digest and print one JSON line (runs in its own process)."""
    files = iter_parquet_files(source)
    fn = versioning._logical_hash if kind == "per_file" else versioning._logical_hash_whole_source
    t0 = time.perf_counter()
    digest, rows, schema = fn(files)
    wall = time.perf_counter() - t0
    out: dict[str, Any] = {
        "kind": kind,
        "files": len(files),
        "logical_hash": digest,
        "rows": rows,
        "schema": schema,
        "wall_s": round(wall, 2),
        "peak_rss_mib": round(_peak_rss_mib(), 1),
        "rss_after_return_mib": _rss_now_mib(),
        "polars": pl.__version__,
    }
    if kind == "per_file":
        out["bytes_hash"] = versioning._bytes_hash(files)
    print(json.dumps(out))


def _run_child(kind: str, source: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, __file__, "--child", kind, str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"kind": kind, "error": f"rc={proc.returncode}", "stderr_tail": proc.stderr[-2000:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _footer_rows(source: Path) -> int:
    return sum(versioning._footer_rows(f) for f in iter_parquet_files(source))


def _git_sha() -> str:
    proc = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "set", type=Path, nargs="?", help="the set root (holds manifest.json and the subtrees)"
    )
    ap.add_argument(
        "--subtree",
        action="append",
        help="a subtree to check (repeatable; default: the manifest's)",
    )
    ap.add_argument("--record", type=Path, help="also write the record here")
    ap.add_argument(
        "--skip-whole-source",
        action="store_true",
        help="compare with the manifest's data_hash only",
    )
    ap.add_argument("--child", nargs=2, metavar=("KIND", "SOURCE"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        _child(a.child[0], Path(a.child[1]))
        return 0
    if a.set is None:
        ap.print_usage(sys.stderr)
        return 2

    root: Path = a.set.expanduser()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    subtrees = a.subtree or sorted(manifest.get("subtrees", {}))
    if not subtrees:
        print(f"no subtrees: {manifest_path} lists none; pass --subtree", file=sys.stderr)
        return 2
    record: dict[str, Any] = {
        "check": "PR-053 per-file logical hash == whole-source (program PR-027 A2)",
        "set": str(root),
        "rux_ml_git_sha": _git_sha(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "subtrees": {},
    }
    ok = True
    for t in subtrees:
        source = root / t
        if not source.is_dir():
            print(f"no subtree directory {source}", file=sys.stderr)
            return 2
        rows = _footer_rows(source)
        est = rows * WHOLE_SOURCE_BYTES_PER_ROW / 2**30
        print(
            f"{t}: {rows:,} rows; whole-source peak estimate {est:.1f} GiB",
            file=sys.stderr,
            flush=True,
        )
        per_file = _run_child("per_file", source)
        entry: dict[str, Any] = {
            "footer_rows": rows,
            "whole_source_estimate_gib": round(est, 1),
            "per_file": per_file,
        }
        checks: dict[str, bool] = {"per_file_ran": "error" not in per_file}
        listed = (manifest.get("ruxml_sidecars") or {}).get(t) or {}
        if listed.get("data_hash") and "error" not in per_file:
            composite = f"{per_file['bytes_hash']}|{per_file['logical_hash']}"
            checks["equals_manifest_data_hash"] = composite == listed["data_hash"]
            entry["manifest_data_hash"] = listed["data_hash"]
        if not a.skip_whole_source:
            whole = _run_child("whole_source", source)
            entry["whole_source"] = whole
            checks["whole_source_ran"] = "error" not in whole
            if checks["per_file_ran"] and checks["whole_source_ran"]:
                checks["equal"] = all(
                    per_file[k] == whole[k] for k in ("logical_hash", "rows", "schema")
                )
        if len(checks) == 1:
            checks["compared_with_something"] = False  # skipped the reference and nothing is listed
        entry["checks"] = checks
        ok = ok and all(checks.values())
        record["subtrees"][t] = entry
    record["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["all_equal"] = ok
    text = json.dumps(record, indent=1, sort_keys=True)
    print(text)
    if a.record:
        a.record.expanduser().write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
