#!/usr/bin/env python3
"""
scripts/rewrite_doc_refs.py

One-shot path-reference rewriter for the hybrid docs-versioning layout
codified by docs/VERSIONING.md (introduced in PR-016).

Migrates references to docs that snapshot into a per-version directory at
each minor/major cut. The mapping table below reflects the v0.2 -> v0.3
cut (PR-036); update it (and run the script) at every subsequent version
cut.

Usage:
    uv run python scripts/rewrite_doc_refs.py --dry-run   # preview changes
    uv run python scripts/rewrite_doc_refs.py             # apply in place

Equivalently via Makefile:
    make rewrite-doc-refs-dry
    make rewrite-doc-refs

The script is idempotent -- re-running after applied is a no-op. The
internal-relative-path fixups use regex with negative lookbehind so a
plain `../prs/` is rewritten to `../../prs/` exactly once even on repeated
runs. Orphaned references (paths that don't appear in the mapping table
when they should) surface naturally at the next cut as unmapped entries
the maintainer must add.

Walks `*.md` and `*.py` files because this project has Python source
modules that reference the temporal docs in module docstrings; the
canonical TS migrator (rux-eth/john/scripts/rewrite-doc-refs.ts) walks
only `*.md`.

OPT_OUT_GLOBS (PR-021 / Q-A6) skip files that contain references which
should stay pinned to the prior version even after the cut:

  - prs/PR-*.md           : historical PR records (memory feedback_pr_spec_historicity)
  - docs/0.0/**           : frozen v0.0 content (intentional self-refs)
  - docs/0.1/**           : frozen v0.1 content (intentional self-refs + history)
  - docs/0.2/**           : frozen v0.2 content (intentional self-refs + history) — added in PR-026
  - docs/0.3/**           : frozen v0.3 content (intentional self-refs + history) — added in PR-036
  - src/**, tests/**      : code citations to specific frozen decisions (D5, D7, ...)

The remaining file set (PROCEDURE-*.md, CLAUDE.md, README.md, docs/SSOT/*)
is mixed. Lines bearing the inline sentinel ``<!-- rewrite-doc-refs:skip-line -->``
are preserved verbatim; everything else gets the PATH_REWRITES treatment.
The sentinel renders as nothing in markdown, and the maintainer adds it
after the apply step to lock historical citations (e.g., "D1-D17 in
DESIGN-log.md") against re-runs.

Run order for a version cut:
    1. (no git mv this cut — docs/<x.y>/ already populated at design time)
    2. Update PATH_REWRITES and OPT_OUT_GLOBS for the new mapping
    3. uv run python scripts/rewrite_doc_refs.py --dry-run   (review)
    4. uv run python scripts/rewrite_doc_refs.py             (apply)
    5. Manually re-pin any historical citations the auto-rewrite missed
    6. git add -A && git commit
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import re

REPO_ROOT = Path.cwd()
SELF_PATH = Path(__file__).resolve()
DRY_RUN = "--dry-run" in sys.argv[1:]

# References to snapshotted files. Applies to every walked file that is
# NOT matched by OPT_OUT_GLOBS below.
PATH_REWRITES: tuple[tuple[str, str], ...] = (
    ("docs/0.2/DESIGN-log.md", "docs/0.3/DESIGN-log.md"),
    ("docs/0.2/RESEARCH-BACKLOG.md", "docs/0.3/RESEARCH-BACKLOG.md"),
    ("docs/0.2/ROADMAP.md", "docs/0.3/ROADMAP.md"),
)

# Files whose references should stay pinned to the prior version. Matched
# against the repo-relative POSIX path via fnmatch (so `*` does NOT cross
# `/` boundaries, but `**` does). See module docstring for rationale.
OPT_OUT_GLOBS: tuple[str, ...] = (
    "prs/PR-*.md",
    "docs/0.0/**",
    "docs/0.1/**",
    "docs/0.2/**",
    "docs/0.3/**",
    "src/**",
    "tests/**",
)

# Lines containing this sentinel are preserved verbatim by the rewriter.
# Renders as nothing in markdown (HTML comment). Used to lock historical
# citations within otherwise rewriteable files.
LINE_SKIP_SENTINEL: str = "<!-- rewrite-doc-refs:skip-line -->"

# Internal-relative-path fixups for files that moved one directory deeper.
# Applied only to the files in MOVED_FILE_PATHS (post-`git mv` locations).
#
# Empty at the v0.0 -> v0.1 cut because docs/0.1/* were populated in place
# during the v0.1 design session (PR #21), not moved via `git mv` in this
# PR. Retained scaffolding for future cuts that do involve a `git mv`.
MOVED_FILE_FIXUPS: tuple[tuple[re.Pattern[str], str], ...] = ()

MOVED_FILE_PATHS: frozenset[str] = frozenset()

IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".basedpyright_cache",
        ".mypy_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "coverage",
    }
)

WALK_SUFFIXES: frozenset[str] = frozenset({".md", ".py"})


def walk_files(root: Path) -> list[Path]:
    out: list[Path] = []
    stack: list[Path] = [root]
    while stack:
        cur = stack.pop()
        for entry in cur.iterdir():
            if entry.name in IGNORE_DIRS:
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.suffix in WALK_SUFFIXES:
                out.append(entry)
    return out


def _rewrite_content(original: str) -> tuple[str, int]:
    """Apply PATH_REWRITES line-by-line, honoring LINE_SKIP_SENTINEL.

    Returns the rewritten text and the count of substitutions made.
    """
    out_lines: list[str] = []
    replacements = 0
    for raw_line in original.splitlines(keepends=True):
        if LINE_SKIP_SENTINEL in raw_line:
            out_lines.append(raw_line)
            continue
        rewritten = raw_line
        for src, dst in PATH_REWRITES:
            count = rewritten.count(src)
            if count > 0:
                rewritten = rewritten.replace(src, dst)
                replacements += count
        out_lines.append(rewritten)
    return "".join(out_lines), replacements


def _is_opted_out(rel_path: str) -> bool:
    """Return True when ``rel_path`` matches any OPT_OUT_GLOBS pattern."""
    for pattern in OPT_OUT_GLOBS:
        # fnmatch treats `*` as anything-but-slash; `**` falls through to
        # plain `*` which is fine because we only use `**` at a path
        # boundary (`docs/0.0/**`) — translate by stripping the trailing
        # `**` and matching the directory prefix.
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if rel_path == prefix or rel_path.startswith(prefix + "/"):
                return True
        elif fnmatch.fnmatchcase(rel_path, pattern):
            return True
    return False


def main() -> int:
    files_scanned = 0
    files_skipped = 0
    files_changed = 0
    total_replacements = 0
    changes: list[tuple[str, int]] = []

    for abs_path in walk_files(REPO_ROOT):
        # Skip self: this script's source contains the PATH_REWRITES literals
        # by necessity, and rewriting them would corrupt the mapping table.
        if abs_path.resolve() == SELF_PATH:
            continue
        rel_path = abs_path.relative_to(REPO_ROOT).as_posix()
        if _is_opted_out(rel_path):
            files_skipped += 1
            continue
        files_scanned += 1
        original = abs_path.read_text(encoding="utf-8")

        # Pass 1: references TO moved files -- line-by-line so LINE_SKIP_SENTINEL
        # can lock historical citations against rewriting (Q-A6 sentinel layer).
        updated, file_replacements = _rewrite_content(original)

        # Pass 2: internal relative-path fixups FROM moved files (regex-based
        # for idempotency -- see MOVED_FILE_FIXUPS docstring).
        if rel_path in MOVED_FILE_PATHS:
            for pattern, replacement in MOVED_FILE_FIXUPS:
                new_text, n = pattern.subn(replacement, updated)
                if n > 0:
                    updated = new_text
                    file_replacements += n

        if updated != original:
            files_changed += 1
            total_replacements += file_replacements
            changes.append((rel_path, file_replacements))
            if not DRY_RUN:
                abs_path.write_text(updated, encoding="utf-8")

    prefix = "[dry-run] " if DRY_RUN else ""
    for file, replacements in sorted(changes, key=lambda c: c[0]):
        print(f"{prefix}{file}: {replacements} replacement(s)")
    print("")
    print(f"{prefix}scanned: {files_scanned} files (opt-out skipped: {files_skipped})")
    verb = "would change" if DRY_RUN else "changed"
    print(f"{prefix}{verb}: {files_changed} file(s); {total_replacements} total replacement(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
