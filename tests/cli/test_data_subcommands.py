"""End-to-end CLI tests for `rux-ml data {hash,version,list}` against synthetic Parquet."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml.cli import app
from tests.conftest import repo_oracle_toml


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
        + repo_oracle_toml()
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


# ---------- PR-047: rux-ml data bridge ----------


def _bridge_set(tmp_path: Path) -> Path:
    src = tmp_path / "fill"
    for day in range(2):
        (src / f"d{day}").mkdir(parents=True)
        pl.DataFrame({"stamp_ms": [day, day + 1], "y__fill_frac": [0.5, 1.0]}).write_parquet(
            src / f"d{day}" / "p.parquet"
        )
    return src


def test_data_bridge_writes_the_sidecar_and_checks_the_manifest(tmp_path: Path) -> None:
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    from typer.testing import CliRunner  # noqa: PLC0415

    from rux_ml.cli import app  # noqa: PLC0415
    from tests.conftest import repo_oracle_toml  # noqa: PLC0415

    src = _bridge_set(tmp_path)
    cfg = tmp_path / "base.toml"
    cfg.write_text(
        f'[data]\nsource_path = "{src}"\ntarget_column = "y__fill_frac"\n' + repo_oracle_toml()
    )
    side = tmp_path / "out" / "fill.ruxml.json"
    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(cfg), "data", "bridge", "--output", str(side)])
    assert result.exit_code == 0, result.output
    record = json.loads(side.read_text())
    assert record["opened_files"] == ["d0/p.parquet", "d1/p.parquet"]
    manifest = {
        "files": [
            {
                "path": f"d{d}/p.parquet",
                "sha256": hashlib.sha256((src / f"d{d}" / "p.parquet").read_bytes()).hexdigest(),
            }
            for d in range(2)
        ],
        "ruxml_data_hash": record["data_hash"],
    }
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps(manifest))
    argv = ["--config", str(cfg), "data", "bridge", "--output", str(side), "--manifest", str(man)]
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert json.loads(side.read_text())["manifest_check"]["equal"] is True
    manifest["files"][0]["sha256"] = "0" * 64
    man.write_text(json.dumps(manifest))
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "data bridge" in result.output
