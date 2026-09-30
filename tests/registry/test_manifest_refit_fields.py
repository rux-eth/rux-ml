"""PR-059 (program PR-028a Q16 / Q4e, Phase 4 item 3): the model manifest records how to
reproduce the evaluated model.

The fields are optional, so a manifest written before them still loads; the manifest stays
``extra="forbid"`` otherwise. They round-trip through ``save_bundle`` / ``load_bundle``
unchanged, the served iteration range included (JSON has no tuple: it is written as a list
and read back as the pair).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from rux_ml.registry.bundle import MANIFEST_FILENAME, load_bundle, save_bundle
from rux_ml.registry.manifest import (
    ManifestSchemaError,
    ModelManifest,
    OosRecord,
    RefitRecord,
    read,
    write,
)
from tests.registry.test_bundle import (  # pyright: ignore[reportPrivateUsage]
    _sample_manifest,
    _trivial_booster,
    _trivial_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

# The trial's fold_meta.json ``oos`` record as rux-ml ``train`` writes it for the walk
# (DG-2's trial: TEST MAE 0.5565, no-underdeduct 0.695), with a key the typed block does
# not name (``gates_path``) that must survive verbatim.
_OOS: dict[str, Any] = {
    "partition": "test",
    "n_rows": 2_811_192,
    "n_scored": 2_811_192,
    "metric": "mae",
    "score": 0.5564869910,
    "gates_sha256": "a" * 64,
    "gates_path": "/x/gates.yaml",
    "honesty": {"no_underdeduct_frac": 0.6949507540, "honest": False},
}


def _with_refit_fields() -> ModelManifest:
    return _sample_manifest().model_copy(
        update={
            "refit": RefitRecord(
                train_rows=13_077_048,
                val_rows=2_771_548,
                best_iteration=98,
                iteration_range=(0, 99),
                device="cuda:0",
                nthread=0,
            ),
            "oos_export_sha256": "1" * 64,
            "oos": OosRecord.model_validate(_OOS),
        }
    )


def test_the_refit_fields_round_trip_through_save_and_load(tmp_path: Path) -> None:
    manifest = _with_refit_fields()
    save_bundle(_trivial_pipeline(), _trivial_booster(), manifest, tmp_path / "bundle")
    _, _, got = load_bundle(tmp_path / "bundle")

    assert got == manifest
    assert got.refit is not None and got.refit.iteration_range == (0, 99)
    assert got.refit.best_iteration == 98 and got.refit.device == "cuda:0"
    assert got.oos_export_sha256 == "1" * 64
    assert got.oos is not None and got.oos.score == pytest.approx(0.5564869910, abs=0)
    # the trial's record verbatim, keys the block does not type included
    assert got.oos.model_dump() == _OOS
    raw = json.loads((tmp_path / "bundle" / MANIFEST_FILENAME).read_text())
    assert raw["refit"]["iteration_range"] == [0, 99]
    assert raw["oos"]["gates_path"] == "/x/gates.yaml"


def test_a_manifest_written_before_the_fields_still_loads(tmp_path: Path) -> None:
    path = tmp_path / MANIFEST_FILENAME
    write(path, _sample_manifest())
    raw = json.loads(path.read_text())
    for key in ("refit", "oos_export_sha256", "oos"):
        raw.pop(key, None)  # an old manifest has none of them
    path.write_text(json.dumps(raw))

    got = read(path)
    assert got.refit is None and got.oos_export_sha256 is None and got.oos is None


def test_the_manifest_still_refuses_an_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / MANIFEST_FILENAME
    write(path, _with_refit_fields())
    raw = json.loads(path.read_text())
    raw["refit"]["served_trees"] = 99  # the refit block is closed too
    path.write_text(json.dumps(raw))
    with pytest.raises(ManifestSchemaError):
        read(path)


def test_the_iteration_range_is_the_best_iteration_plus_one() -> None:
    with pytest.raises(ValueError, match="iteration_range"):
        RefitRecord(
            train_rows=1, val_rows=1, best_iteration=98, iteration_range=(0, 100),
            device="cpu", nthread=0,
        )  # fmt: skip
    # no early stopping: XGBoost's (0, 0) — every tree
    assert RefitRecord(
        train_rows=1, val_rows=1, best_iteration=None, iteration_range=(0, 0),
        device="cpu", nthread=0,
    ).iteration_range == (0, 0)  # fmt: skip
