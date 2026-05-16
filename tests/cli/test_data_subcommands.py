"""End-to-end CLI tests for `rux-ml data {hash,version,list}` against synthetic Parquet."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml.cli import app


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A workdir with a tiny config that points cas/manifests under tmp_path."""
    src_parquet = tmp_path / "src.parquet"
    pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 0]}).write_parquet(src_parquet)

    config = tmp_path / "base.toml"
    config.write_text(
        f"""
        [data]
        cas_root = "{tmp_path}/cas"
        manifests_root = "{tmp_path}/manifests"
        """
    )
    return tmp_path


def _argv(workdir: Path, *args: str) -> list[str]:
    return ["--config", str(workdir / "base.toml"), "data", *args]


def test_data_hash_prints_json(runner: CliRunner, workdir: Path) -> None:
    src = workdir / "src.parquet"
    result = runner.invoke(app, _argv(workdir, "hash", str(src)))
    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert set(parsed) == {"bytes_hash", "logical_hash", "row_count", "schema"}
    assert parsed["row_count"] == 5


def test_data_version_then_list(runner: CliRunner, workdir: Path) -> None:
    src = workdir / "src.parquet"

    snap = runner.invoke(app, _argv(workdir, "version", "mydata", str(src)))
    assert snap.exit_code == 0, snap.stderr
    assert "snapshotted mydata" in snap.stdout

    lst = runner.invoke(app, _argv(workdir, "list"))
    assert lst.exit_code == 0, lst.stderr
    assert "mydata" in lst.stdout
    assert "rows=5" in lst.stdout


def test_data_list_empty(runner: CliRunner, workdir: Path) -> None:
    result = runner.invoke(app, _argv(workdir, "list"))
    assert result.exit_code == 0
    assert "no manifests found" in result.stdout


def test_data_list_filters_by_name(runner: CliRunner, workdir: Path) -> None:
    src = workdir / "src.parquet"
    runner.invoke(app, _argv(workdir, "version", "alpha", str(src)))
    runner.invoke(app, _argv(workdir, "version", "beta", str(src)))

    result = runner.invoke(app, _argv(workdir, "list", "--name", "alpha"))
    assert result.exit_code == 0
    assert "alpha" in result.stdout
    assert "beta" not in result.stdout
