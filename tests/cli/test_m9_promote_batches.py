"""PR-055 (program PR-027 R3; the operator's ruling of 2026-09-28): the [m9] promote re-fit
runs on the per-day batch path.

``promote`` re-fits the trial's model before it bundles it. At 156 days the in-memory
re-fit (the whole source collected, the partitions as pandas frames) exceeds the 20 GiB
scope that the per-day batch fit ``rux-ml train`` runs in (PR-054) fits inside. So the
re-fit takes the batch path, with the same membership rule and seeds, and its bundle must
predict **bit-identically** to the in-memory re-fit's. Promote's refusals (PR-050's
``data_hash`` equality, PR-051's harness manifest) run before the re-fit and are unchanged
(their own tests stay green).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import optuna
import polars as pl
import pytest
import xgboost as xgb

from rux_ml._internal.seeds import SeedBag, make_seed_bag
from rux_ml.cli import app
from rux_ml.data import check_feature_labels, load_parquet, make_splits, materialize
from rux_ml.features import cardinalities_from, make_features
from rux_ml.registry import promote as promote_module
from rux_ml.registry.bundle import load_bundle
from rux_ml.registry.paths import version_dir
from rux_ml.runs.provenance import data_hashes_and_files
from rux_ml.training import make_trainer
from tests.cli.conftest import m9_argv
from tests.cli.test_m9_batches import STUDIES, TARGET, _cfg  # pyright: ignore[reportPrivateUsage]
from tests.conftest import m9_gates_overrides

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.pipeline import Pipeline
    from typer.testing import CliRunner

    from rux_ml.config import RuxMLConfig


def _refuse(*_a: object, **_k: object) -> None:
    msg = "the [m9] promote re-fit collected the whole source"
    raise AssertionError(msg)


def _in_memory_refit(cfg: RuxMLConfig, bag: SeedBag) -> tuple[Pipeline, xgb.Booster]:
    """The pre-PR-055 ``promote._refit``, verbatim: whole frames, the sklearn wrapper."""
    assert cfg.data.source_path is not None and cfg.data.target_column is not None
    check_feature_labels(cfg)
    df = materialize(load_parquet(cfg.data.source_path, oracle=cfg.data.oracle))
    splits = make_splits(cfg, df, seed=bag.split_seed)
    x_train = splits["train"].drop(cfg.data.target_column)
    y_train = splits["train"][cfg.data.target_column]
    x_val = splits["val"].drop(cfg.data.target_column)
    y_val = splits["val"][cfg.data.target_column]
    cards = cardinalities_from(x_train, cfg.features.spec.categorical_columns)
    pipeline = make_features(cfg.features, cardinalities=cards if cards else None)
    pipeline.fit(x_train, y_train.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    x_train_t = cast("pl.DataFrame", pipeline.transform(x_train))  # pyright: ignore[reportUnknownMemberType]
    x_val_t = cast("pl.DataFrame", pipeline.transform(x_val))  # pyright: ignore[reportUnknownMemberType]
    trainer = cast("Any", make_trainer(cfg.training, seed=bag.xgb_seed))
    trainer.fit(
        x_train_t.to_pandas(),
        y_train.to_numpy(),
        eval_set=[(x_val_t.to_pandas(), y_val.to_numpy())],
        verbose=False,
    )
    return pipeline, cast("xgb.Booster", trainer.get_booster())


def _bundle_predictions(pipeline: Pipeline, booster: xgb.Booster, frame: pl.DataFrame) -> str:
    """The registry scorer's route (``scorer._BoosterShim``): the pipeline, then
    ``booster.predict`` on a categorical-enabled ``DMatrix`` — as a sha256."""
    x = cast("pl.DataFrame", pipeline.transform(frame.drop(TARGET)))  # pyright: ignore[reportUnknownMemberType]
    pred = np.asarray(booster.predict(xgb.DMatrix(x.to_pandas(), enable_categorical=True)))
    return hashlib.sha256(np.ascontiguousarray(pred, dtype=np.float32).tobytes()).hexdigest()


@pytest.mark.parametrize("regime", sorted(STUDIES))
def test_the_batch_refit_predicts_bit_identically_to_the_in_memory_refit(
    regime: str, c6_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(c6_set, regime)  # subsample = colsample_bytree = 0.8: seeds and row order matter
    assert cfg.data.source_path is not None
    bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=0)
    want_pipeline, want_booster = _in_memory_refit(cfg, bag)
    frame = materialize(load_parquet(cfg.data.source_path, oracle=cfg.data.oracle))
    want = _bundle_predictions(want_pipeline, want_booster, frame)

    monkeypatch.setattr(promote_module, "materialize", _refuse)
    _, hashed = data_hashes_and_files(cfg.data.source_path, oracle=cfg.data.oracle)  # PR-059: Q6
    got_pipeline, got_booster, _ = promote_module._refit(cfg, bag=bag, hashed_files=hashed)  # pyright: ignore[reportPrivateUsage]

    assert _bundle_predictions(got_pipeline, got_booster, frame) == want
    assert got_booster.best_iteration == want_booster.best_iteration
    assert got_booster.num_boosted_rounds() == want_booster.num_boosted_rounds()
    assert list(got_pipeline[0].feature_names_in_) == list(want_pipeline[0].feature_names_in_)
    # sensitivity: another trial's seeds give other predictions, so equality has teeth
    other_pipeline, other_booster = _in_memory_refit(
        cfg, make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=1)
    )
    assert _bundle_predictions(other_pipeline, other_booster, frame) != want


def test_an_m9_promote_never_materialises_the_whole_source(
    runner: CliRunner,
    c6_set: Path,
    tmp_path: Path,
    signed_m9_gates: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = m9_argv(tmp_path, c6_set, "m9_fill_frac", *m9_gates_overrides(signed_m9_gates))
    trained = runner.invoke(app, [*base, "train"], catch_exceptions=False)
    assert trained.exit_code == 0, trained.output
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)

    monkeypatch.setattr(promote_module, "materialize", _refuse)
    promoted = runner.invoke(
        app,
        [*base, "registry", "promote", "--problem", "m9_fill_frac",
         "--study", summary.study_name, "--trial", "0"],
        catch_exceptions=False,
    )  # fmt: skip
    assert promoted.exit_code == 0, promoted.output
    version = promoted.output.split("promoted: m9_fill_frac@", 1)[1].split()[0]
    pipeline, booster, manifest = load_bundle(
        version_dir(tmp_path / "registry", "m9_fill_frac", version)
    )
    assert manifest.harness_manifest is not None  # PR-051's record, unchanged
    assert booster.num_boosted_rounds() > 0
    assert pipeline is not None
