"""Root-app smoke tests: help, version, verb-group wiring."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rux_ml import __version__
from rux_ml.cli import app

VERB_GROUPS = ("data", "train", "tune", "runs", "registry")


def test_help_lists_all_verb_groups(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in VERB_GROUPS:
        assert verb in result.stdout


def test_help_includes_program_description(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert "rux-ml" in result.stdout


def test_version_flag(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help(runner: CliRunner) -> None:
    # ``no_args_is_help=True`` on the root app
    result = runner.invoke(app, [])
    # Typer's no-args-is-help exits with code 2 in Click 8.x; both 0 and 2 are acceptable.
    assert result.exit_code in (0, 2)
    assert "Usage" in result.stdout or "Usage" in (result.stderr or "")


@pytest.mark.parametrize("group", VERB_GROUPS)
def test_each_verb_group_help_works(runner: CliRunner, group: str) -> None:
    if group == "train":
        # train is a leaf command, not a sub-app; --help shows command help, not group help
        result = runner.invoke(app, [group, "--help"])
        assert result.exit_code == 0
        return
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0
    assert group in result.stdout.lower() or "Usage" in result.stdout
