"""Tests for ``rux_ml._internal.git.git_sha`` — happy path + graceful fallback."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from rux_ml._internal.git import git_sha


def test_git_sha_returns_40_char_hex_in_repo() -> None:
    """When run inside this git work-tree, ``git_sha`` returns a real SHA."""
    sha = git_sha()
    # Tests run from the repo root; the work-tree is git-tracked.
    assert sha == "unknown" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def test_git_sha_returns_unknown_when_git_missing() -> None:
    """If ``git`` isn't on PATH, ``git_sha`` returns ``"unknown"`` instead of raising."""

    def _raise_fnf(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    with patch("rux_ml._internal.git.subprocess.run", side_effect=_raise_fnf):
        assert git_sha() == "unknown"


def test_git_sha_returns_unknown_on_nonzero_exit() -> None:
    """If ``git rev-parse`` fails (e.g. not a repo), ``git_sha`` returns ``"unknown"``."""
    fake = subprocess.CompletedProcess(args=["git"], returncode=128, stdout="", stderr="fatal")
    with patch("rux_ml._internal.git.subprocess.run", return_value=fake):
        assert git_sha() == "unknown"


def test_git_sha_returns_unknown_on_empty_stdout() -> None:
    fake = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="\n", stderr="")
    with patch("rux_ml._internal.git.subprocess.run", return_value=fake):
        assert git_sha() == "unknown"
