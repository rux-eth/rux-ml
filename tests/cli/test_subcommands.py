"""Smoke-invoke every no-op subcommand. Confirms wiring + GlobalOptions plumbing."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rux_ml.cli import app

# (argv, expected substring in stderr noting which PR will implement it)
SUBCOMMANDS: list[tuple[list[str], str]] = [
    # data verbs landed in PR-004, `train` in PR-006, and `tune` in PR-007 —
    # they no longer print "not yet implemented". Covered by
    # tests/cli/test_data_subcommands.py / test_train_subcommand.py /
    # test_tune_subcommands.py instead.
    (["runs", "list"], "PR-009"),
    (["runs", "show", "1"], "PR-009"),
    (["runs", "compare", "1", "2"], "PR-009"),
    (
        ["registry", "promote", "--problem", "churn_v1", "--study", "s", "--trial", "1"],
        "PR-010",
    ),
    (["registry", "list"], "PR-010"),
    (
        ["registry", "rollback", "--problem", "churn_v1", "--to", "v_2026_05_14_a8f"],
        "PR-010",
    ),
]


@pytest.mark.parametrize(("argv", "expected_pr"), SUBCOMMANDS)
def test_subcommand_is_wired(runner: CliRunner, argv: list[str], expected_pr: str) -> None:
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, f"argv={argv}, stderr={result.stderr}"
    combined = result.stdout + (result.stderr or "")
    assert "not yet implemented" in combined
    assert expected_pr in combined
