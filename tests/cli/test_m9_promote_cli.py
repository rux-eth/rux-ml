"""End-to-end: an M9 model records its harness set's manifest id, or promote refuses (PR-051).

rux-capital program v0.3 D45 #6 / ACCEPTANCE C11: "rux-ml's model manifest carries the
harness manifest id". The repo's real ``m9_walk_bp`` problem is trained and promoted
through the real CLI on the synthetic C6 set, which carries the materializer's
``manifest.json`` and views like the real one (program PR-024).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import optuna
from typer.testing import CliRunner

from rux_ml.cli import app
from rux_ml.registry.manifest import read as read_manifest
from rux_ml.registry.paths import version_dir
from tests.cli.conftest import m9_argv
from tests.conftest import m9_gates_overrides

if TYPE_CHECKING:
    from pathlib import Path


def test_m9_promote_records_the_harness_manifest_id_and_refuses_without_it(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    base = m9_argv(tmp_path, c6_set, "m9_walk_bp", *m9_gates_overrides(signed_m9_gates))
    trained = runner.invoke(app, [*base, "train"], catch_exceptions=False)
    assert trained.exit_code == 0, trained.output
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    promote = [
        *base, "registry", "promote", "--problem", "m9_walk_bp",
        "--study", summary.study_name, "--trial", "0",
    ]  # fmt: skip

    view = c6_set / "walk.manifest.json"
    moved = view.rename(tmp_path / "walk.manifest.json")
    refused = runner.invoke(app, promote)
    assert refused.exit_code == 2, refused.output
    assert "harness manifest" in refused.output
    assert not (tmp_path / "registry" / "m9_walk_bp").exists()

    moved.rename(view)
    promoted = runner.invoke(app, promote, catch_exceptions=False)
    assert promoted.exit_code == 0, promoted.output
    champion = json.loads((tmp_path / "registry" / "m9_walk_bp" / "champion.json").read_text())
    vdir = version_dir(tmp_path / "registry", "m9_walk_bp", champion["version"])
    ref = read_manifest(vdir / "manifest.json").harness_manifest
    assert ref is not None
    assert ref.manifest_id == hashlib.sha256((c6_set / "manifest.json").read_bytes()).hexdigest()
    assert (ref.set_name, ref.subtree) == (c6_set.name, "walk")
