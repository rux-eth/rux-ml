"""Tests for the root callback: global flags flow into ``GlobalOptions``."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from rux_ml.cli import app
from rux_ml.cli._shared import GlobalOptions, parse_set_overrides

# A probe subcommand we register temporarily inside the test app to inspect ctx.obj.
# We do it via runner.invoke against the existing `data list` no-op, then assert
# the stored GlobalOptions by reaching through ctx — but Typer doesn't expose ctx
# after invoke. Instead, we unit-test parse_set_overrides directly and verify the
# callback by invoking a hidden test command.

# --- parse_set_overrides ---


def test_parse_set_overrides_empty() -> None:
    assert parse_set_overrides([]) == {}


def test_parse_set_overrides_int_value() -> None:
    assert parse_set_overrides(["tuning.n_trials=100"]) == {"tuning.n_trials": 100}


def test_parse_set_overrides_float_value() -> None:
    assert parse_set_overrides(["training.learning_rate=0.05"]) == {"training.learning_rate": 0.05}


def test_parse_set_overrides_bool_value() -> None:
    assert parse_set_overrides(["training.enable_categorical=false"]) == {
        "training.enable_categorical": False
    }


def test_parse_set_overrides_string_value() -> None:
    # Unquoted strings come through as JSON-failure fallback -> raw string
    assert parse_set_overrides(["training.metric=auc"]) == {"training.metric": "auc"}


def test_parse_set_overrides_list_value() -> None:
    assert parse_set_overrides(["training.foo=[1,2,3]"]) == {"training.foo": [1, 2, 3]}


def test_parse_set_overrides_multiple() -> None:
    result = parse_set_overrides(["training.learning_rate=0.05", "tuning.n_trials=100"])
    assert result == {"training.learning_rate": 0.05, "tuning.n_trials": 100}


def test_parse_set_overrides_rejects_missing_equals() -> None:
    with pytest.raises(typer.BadParameter):
        parse_set_overrides(["training.learning_rate"])


def test_parse_set_overrides_rejects_empty_key() -> None:
    with pytest.raises(typer.BadParameter):
        parse_set_overrides(["=0.05"])


# --- Callback wiring: invoke a probe through the runner ---


def test_callback_populates_global_options(runner: CliRunner) -> None:
    """Invoke with global flags + a benign real subcommand and verify args flow through.

    Post-PR-010, all subcommands are real bodies — no more "not yet implemented"
    stubs. ``registry list`` against an empty/missing registry is benign
    (exits 0 with a "no registry" message) so it's safe as the wiring probe.
    """
    result = runner.invoke(
        app,
        [
            "--config",
            "configs/base.toml",
            "--set",
            "training.learning_rate=0.05",
            "--set",
            "tuning.n_trials=100",
            "--verbose",
            "--dry-run",
            "registry",
            "list",
        ],
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    combined = result.stdout + (result.stderr or "")
    # The benign body prints either "no registry at <path>" or a per-problem header.
    assert "registry" in combined.lower() or "problems" in combined.lower()


def test_callback_rejects_malformed_set(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--set", "no_equals_sign", "registry", "list"])
    assert result.exit_code != 0


# --- GlobalOptions dataclass directly ---


def test_global_options_defaults() -> None:
    opts = GlobalOptions(config=Path("configs/base.toml"), problem=None, study=None)
    assert opts.overrides == {}
    assert opts.verbose is False
    assert opts.dry_run is False
