#!/usr/bin/env python3
"""
scripts/rewrite_doc_refs.py

One-shot path-reference rewriter for the hybrid docs-versioning layout
codified by docs/VERSIONING.md (introduced in PR-016).

Migrates references to docs that move into a per-version directory at each
minor/major cut. The mapping table below reflects the v0.0 -> docs/0.0/ cut;
update it (and run the script) at every subsequent version cut.

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

Run order for a version cut:
    1. git mv docs/<file>.md docs/<x.y>/<file>.md      (per file moved)
    2. uv run python scripts/rewrite_doc_refs.py --dry-run   (review)
    3. uv run python scripts/rewrite_doc_refs.py             (apply)
    4. git add -A && git commit
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
SELF_PATH = Path(__file__).resolve()
DRY_RUN = "--dry-run" in sys.argv[1:]

# References to moved files. Applies to every walked file.
PATH_REWRITES: tuple[tuple[str, str], ...] = (
    ("docs/DESIGN-log.md", "docs/0.0/DESIGN-log.md"),
    ("docs/RESEARCH-BACKLOG.md", "docs/0.0/RESEARCH-BACKLOG.md"),
    ("docs/ROADMAP.md", "docs/0.0/ROADMAP.md"),
)

# Internal-relative-path fixups for files that moved one directory deeper.
# Applied only to the files in MOVED_FILE_PATHS (post-`git mv` locations).
#
# Uses regex with negative lookbehind so the rewrite is idempotent --
# `../prs/` matches only when NOT preceded by `../`, so `../../prs/`
# (the post-rewrite form) does not match a second time. Plain string
# replace would corrupt to `../../../prs/` on re-run.
MOVED_FILE_FIXUPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!\.\./)\.\./prs/"), "../../prs/"),
)

MOVED_FILE_PATHS: frozenset[str] = frozenset(
    {
        "docs/0.0/DESIGN-log.md",
        "docs/0.0/RESEARCH-BACKLOG.md",
        "docs/0.0/ROADMAP.md",
    }
)

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


def main() -> int:
    files_scanned = 0
    files_changed = 0
    total_replacements = 0
    changes: list[tuple[str, int]] = []

    for abs_path in walk_files(REPO_ROOT):
        # Skip self: this script's source contains the PATH_REWRITES literals
        # by necessity, and rewriting them would corrupt the mapping table.
        if abs_path.resolve() == SELF_PATH:
            continue
        files_scanned += 1
        rel_path = abs_path.relative_to(REPO_ROOT).as_posix()
        original = abs_path.read_text(encoding="utf-8")
        updated = original
        file_replacements = 0

        # Pass 1: references TO moved files -- applies to every walked file.
        for src, dst in PATH_REWRITES:
            count = updated.count(src)
            if count > 0:
                updated = updated.replace(src, dst)
                file_replacements += count

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
    print(f"{prefix}scanned: {files_scanned} files")
    verb = "would change" if DRY_RUN else "changed"
    print(f"{prefix}{verb}: {files_changed} file(s); {total_replacements} total replacement(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
