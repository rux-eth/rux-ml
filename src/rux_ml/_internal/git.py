"""Git provenance helpers for the per-trial ``user_attrs`` set.

PR-006 records ``git_sha`` on every Optuna trial so a trial can be replayed
against the exact code that produced it. The implementation shells out to
``git`` rather than importing GitPython — keeps the dependency footprint small
and matches what the surrounding ``_internal`` modules do (subprocess for
side-effect work, stdlib for everything else).
"""

from __future__ import annotations

import subprocess

_UNKNOWN = "unknown"


def git_sha() -> str:
    """Return the current commit SHA, or ``"unknown"`` if git can't resolve one.

    Failure modes that resolve to ``"unknown"``:
    - The process isn't inside a git work-tree (e.g. installed wheel in a CI image).
    - ``git`` isn't on ``PATH``.
    - The repo has no commits yet.

    The return value is recorded verbatim in Optuna ``user_attrs``; downstream
    promotion code can refuse to promote trials whose ``git_sha == "unknown"``
    (per the provenance-triple constraint in ``docs/CONSTRAINTS.md``).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return _UNKNOWN
    if result.returncode != 0:
        return _UNKNOWN
    sha = result.stdout.strip()
    return sha or _UNKNOWN
