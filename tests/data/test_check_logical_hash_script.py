"""PR-053: the deferred real-set check (``scripts/check_logical_hash_per_file.py``) runs end to end
on a small harness-shaped set: equal exits 0, a listed ``data_hash`` that differs exits 1, a set
with no subtrees exits 2."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import polars as pl

from rux_ml.data import versioning
from rux_ml.data.loaders import iter_parquet_files

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_logical_hash_per_file.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_logical_hash_per_file", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set(root: Path, listed: str | None) -> Path:
    for t in ("fill", "walk"):
        (root / t).mkdir(parents=True)
        for d in range(3):
            rows = np.arange(d * 100, (d + 1) * 100)
            pl.DataFrame({"x": rows, "y": rows * 0.5}).write_parquet(root / t / f"{d}.parquet")
    files = iter_parquet_files(root / "fill")
    real = f"{versioning._bytes_hash(files)}|{versioning._logical_hash_whole_source(files)[0]}"
    manifest = {
        "subtrees": {"fill": {}, "walk": {}},
        "ruxml_sidecars": {"fill": {"data_hash": listed or real}, "walk": None},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_the_script_passes_on_an_equal_set_and_records_both_digests(tmp_path: Path) -> None:
    root = _set(tmp_path / "set", listed=None)
    record = tmp_path / "record.json"
    assert _script().main([str(root), "--record", str(record)]) == 0
    got = json.loads(record.read_text())
    assert got["all_equal"] is True
    fill = got["subtrees"]["fill"]
    assert fill["checks"] == {
        "per_file_ran": True,
        "equals_manifest_data_hash": True,
        "whole_source_ran": True,
        "equal": True,
    }
    assert fill["per_file"]["rows"] == fill["whole_source"]["rows"] == 300
    assert got["subtrees"]["walk"]["checks"]["equal"] is True


def test_the_script_fails_when_the_listed_data_hash_differs(tmp_path: Path) -> None:
    root = _set(tmp_path / "set", listed="not|this")
    assert _script().main([str(root), "--subtree", "fill", "--skip-whole-source"]) == 1


def test_the_script_refuses_a_set_without_subtrees(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert _script().main([str(tmp_path / "empty")]) == 2
