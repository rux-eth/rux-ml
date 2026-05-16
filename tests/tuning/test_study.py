"""Tests for ``create_or_load`` study wrapper."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
from optuna.pruners import NopPruner
from optuna.samplers import RandomSampler

from rux_ml.tuning import create_or_load


def test_create_or_load_creates_sqlite_db(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    study = create_or_load(
        name="t1",
        storage=storage,
        sampler=RandomSampler(seed=0),
        pruner=NopPruner(),
        direction="maximize",
    )
    assert study.study_name == "t1"
    # Parent dir was created (otherwise SQLite would have failed).
    assert (tmp_path / "studies").exists()


def test_create_or_load_idempotent_on_existing_study(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    a = create_or_load(
        name="t2", storage=storage, sampler=RandomSampler(seed=0),
        pruner=NopPruner(), direction="maximize",
    )
    a.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=2)
    # Reload — should not raise, should preserve trials.
    b = create_or_load(
        name="t2", storage=storage, sampler=RandomSampler(seed=0),
        pruner=NopPruner(), direction="maximize",
    )
    assert len(b.trials) == 2
    assert b.study_name == a.study_name


def test_create_or_load_load_if_exists_false_raises_on_collision(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    create_or_load(
        name="t3", storage=storage, sampler=RandomSampler(seed=0),
        pruner=NopPruner(), direction="maximize",
    )
    with pytest.raises((optuna.exceptions.DuplicatedStudyError, ValueError)):
        create_or_load(
            name="t3", storage=storage, sampler=RandomSampler(seed=0),
            pruner=NopPruner(), direction="maximize",
            load_if_exists=False,
        )
