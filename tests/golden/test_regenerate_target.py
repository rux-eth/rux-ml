"""``make regenerate-golden`` parses its own flag (per PR-057).

``--regenerate-golden`` is registered by ``tests/golden/conftest.py``, and pytest
reads a conftest's ``pytest_addoption`` before parsing the command line only when
that conftest is an initial one: ``tests/golden`` must be on the command line.
Without it the recipe dies with "unrecognized arguments" and regenerates nothing
(found on the desktop, 2026-09-29). Not golden-marked: this runs in the default
suite and fits nothing (collect only).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _recipe(target: str) -> list[str]:
    """The first recipe line of a Makefile ``target``, split into arguments."""
    lines = (REPO_ROOT / "Makefile").read_text().splitlines()
    return shlex.split(lines[lines.index(f"{target}:") + 1].strip())


def test_make_regenerate_golden_parses_its_flag() -> None:
    """The recipe's pytest arguments collect cleanly (collect only: nothing is fit or written)."""
    cmd = _recipe("regenerate-golden")
    assert cmd[:3] == ["uv", "run", "pytest"], cmd
    assert "--regenerate-golden" in cmd, cmd
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *cmd[3:],
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
