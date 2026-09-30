"""PR-059 (program PR-028a Q16 / Q4e / Q6, Phase 4 item 3): an [m9] promote records how to
reproduce the evaluated model, and refuses a re-fit whose loader would read other files
than the hash covered.

- The re-fit's train / val rows equal the trial's partition plan (``split_definition``).
- ``best_iteration`` and the served ``iteration_range`` equal the trial's; XGBoost 3.2's
  ``Booster.predict`` default uses every tree (program DG-2), so a consumer pins the range.
- The fit's device / nthread, the trial's ``oos_export_sha256`` and its TEST record (the
  manifest's ``metric_value`` is the Optuna value, the VAL score) are the trial's own.
- Q6: the batch loader's file list equals the list the data hash covered, else refused.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import optuna
import pytest

from rux_ml._internal.seeds import make_seed_bag
from rux_ml.cli import app
from rux_ml.registry import promote as promote_module
from rux_ml.registry.bundle import load_bundle
from rux_ml.registry.paths import version_dir
from rux_ml.runs import load_run
from rux_ml.runs.provenance import data_hashes, data_hashes_and_files
from tests.cli.conftest import m9_argv
from tests.cli.test_m9_batches import _cfg  # pyright: ignore[reportPrivateUsage]
from tests.cli.test_m9_train_cli import _fold_meta  # pyright: ignore[reportPrivateUsage]
from tests.conftest import m9_gates_overrides

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import CliRunner

PROBLEM = "m9_markout_bp"  # a regression with the honesty test: the TEST record has it


def test_an_m9_promote_records_the_refit_the_range_and_the_trials_test_record(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    base = m9_argv(tmp_path, c6_set, PROBLEM, *m9_gates_overrides(signed_m9_gates))
    trained = runner.invoke(app, [*base, "train"], catch_exceptions=False)
    assert trained.exit_code == 0, trained.output
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    promoted = runner.invoke(
        app,
        [*base, "registry", "promote", "--problem", PROBLEM,
         "--study", summary.study_name, "--trial", "0"],
        catch_exceptions=False,
    )  # fmt: skip
    assert promoted.exit_code == 0, promoted.output
    version = promoted.output.split(f"promoted: {PROBLEM}@", 1)[1].split()[0]
    _, booster, manifest = load_bundle(version_dir(tmp_path / "registry", PROBLEM, version))

    run = load_run(storage, summary.study_name, 0)
    attrs, meta = run.attrs, _fold_meta(tmp_path, PROBLEM)
    assert attrs is not None
    rows = meta["split_definition"]["rows"]
    refit = manifest.refit
    assert refit is not None
    assert refit.train_rows == rows["train"] == meta["row_count"]
    assert refit.val_rows == rows["val"]
    assert attrs.best_iteration is not None  # early stopping ran, so the range has teeth
    assert refit.best_iteration == attrs.best_iteration == booster.best_iteration
    assert refit.iteration_range == (0, attrs.best_iteration + 1)
    assert (refit.device, refit.nthread) == (attrs.booster_device, attrs.booster_nthread)
    assert (refit.device, refit.nthread) == (meta["booster"]["device"], meta["booster"]["nthread"])
    assert attrs.oos_export_sha256 is not None
    assert manifest.oos_export_sha256 == attrs.oos_export_sha256
    assert manifest.oos_export_sha256 == meta["oos_export"]["sha256"]
    # the TEST record verbatim; metric_value stays the Optuna (VAL) value
    assert manifest.oos is not None and manifest.oos.model_dump() == meta["oos"]
    assert manifest.oos.partition == "test" and manifest.oos.n_rows == rows["test"]
    assert manifest.metric_value == run.value
    assert manifest.oos.honesty is not None and "forms" in manifest.oos.honesty


def test_a_non_m9_promote_without_a_test_record_leaves_the_block_empty() -> None:
    """A trial with no fold_meta.json ``oos`` record (every fit off the [m9] path) promotes
    with ``oos = None``: the field is optional, never invented."""
    assert promote_module.trial_oos_record([{"fold_idx": 0, "row_count": 3}]) is None
    assert promote_module.trial_oos_record([]) is None
    with pytest.raises(ValueError, match="more than one"):
        promote_module.trial_oos_record([{"oos": {"partition": "test", "n_rows": 1}}] * 2)


@pytest.mark.parametrize("change", ["file_added", "file_dropped_from_hash"])
def test_the_refit_refuses_a_file_set_the_hash_did_not_cover(
    c6_set: Path, tmp_path: Path, change: str
) -> None:
    root = tmp_path / "copy" / "set"
    shutil.copytree(c6_set, root)
    cfg = _cfg(root, "time_ordered")
    assert cfg.data.source_path is not None
    hashes, files = data_hashes_and_files(cfg.data.source_path, oracle=cfg.data.oracle)
    assert hashes == data_hashes(cfg.data.source_path, oracle=cfg.data.oracle)
    assert files and all(f.parent == cfg.data.source_path for f in files)
    bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=0)

    if change == "file_added":  # a day file lands between the hash and the re-fit
        extra = cfg.data.source_path / "2026-01-01.parquet"
        shutil.copy(files[0], extra)
        named = extra.name
    else:
        named, files = files[-1].name, files[:-1]
    with pytest.raises(promote_module.PromoteFileSetError, match=named):
        promote_module._refit(cfg, bag=bag, hashed_files=files)  # pyright: ignore[reportPrivateUsage]
