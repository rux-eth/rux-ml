"""The bridge's equality test on the C6 set's real manifest shape, through the CLI (PR-052).

Program PR-024's materializer writes rux-ml's ``data_hash`` into the set manifest at
``ruxml_sidecars.<subtree>.data_hash`` once rux-ml's sidecar exists (``null`` before),
and a per-subtree view with the file list only. PR-047's check read a top-level
``ruxml_data_hash`` that nothing writes, so against the view it passed on the file list
alone. The flow here is the real one: write the sidecar, the harness lists it, check.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from rux_ml.cli import app
from tests.cli.conftest import m9_argv
from tests.conftest import m9_gates_overrides, write_harness_manifest

if TYPE_CHECKING:
    from pathlib import Path


def _recorded_error(tmp: Path) -> str:
    """The refusal as the CLI records it in the check's sidecar (the console wraps it)."""
    check = json.loads((tmp / "check.json").read_text())["manifest_check"]
    assert check["equal"] is False
    return str(check["error"])


def test_the_bridge_checks_the_data_hash_the_harness_lists(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    base = m9_argv(tmp_path, c6_set, "m9_walk_bp", *m9_gates_overrides(signed_m9_gates))
    sidecar = c6_set / "walk.ruxml-sidecar.json"
    wrote = runner.invoke(app, [*base, "data", "bridge", "--output", str(sidecar)])
    assert wrote.exit_code == 0, wrote.output
    data_hash = json.loads(sidecar.read_text())["data_hash"]
    check = [*base, "data", "bridge", "--output", str(tmp_path / "check.json"), "--manifest"]

    # The per-subtree view carries the file list only: no longer a pass on files alone.
    on_view = runner.invoke(app, [*check, str(c6_set / "walk.manifest.json")])
    assert on_view.exit_code == 2, on_view.output

    # The set manifest before the harness lists the sidecar (ruxml_sidecars.walk = null).
    unlisted = runner.invoke(app, [*check, str(c6_set / "manifest.json")])
    assert unlisted.exit_code == 2, unlisted.output
    assert "no rux-ml data_hash for walk/" in _recorded_error(tmp_path)

    # The harness lists a data_hash that is not rux-ml's.
    write_harness_manifest(c6_set, ["fill", "walk"], ruxml_data_hash={"walk": "0|0"})
    other = runner.invoke(app, [*check, str(c6_set / "manifest.json")])
    assert other.exit_code == 2, other.output
    assert "ruxml_sidecars.walk.data_hash 0|0 != rux-ml data_hash" in _recorded_error(tmp_path)

    # The harness lists rux-ml's sidecar: the equality test holds, data_hash included.
    write_harness_manifest(c6_set, ["fill", "walk"], ruxml_data_hash={"walk": data_hash})
    equal = runner.invoke(app, [*check, str(c6_set / "manifest.json")])
    assert equal.exit_code == 0, equal.output
    record = json.loads((tmp_path / "check.json").read_text())["manifest_check"]
    assert record == {"subtree": "walk", "files": 3, "data_hash": data_hash, "equal": True}
