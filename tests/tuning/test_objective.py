"""Tests for ``walk_search_space`` + ``build_objective``."""

from __future__ import annotations

import optuna
import pytest
from optuna.pruners import NopPruner
from optuna.samplers import RandomSampler

from rux_ml.config import (
    CatSpec,
    DataConfig,
    FloatSpec,
    IntSpec,
    KFoldCV,
    RuxMLConfig,
    SearchSpec,
    StratifiedKFoldCV,
)
from rux_ml.tuning import build_objective, walk_search_space

# ---------- walk_search_space ----------


def test_walk_search_space_translates_each_variant() -> None:
    space: dict[str, SearchSpec] = {
        "training.learning_rate": FloatSpec(low=0.01, high=0.3, log=True),
        "training.max_depth": IntSpec(low=3, high=8),
        "training.colsample_bytree": CatSpec(choices=[0.8, 1.0]),
    }
    # Use Optuna's create_study + study.ask to get a real Trial we can inspect.
    study = optuna.create_study(sampler=RandomSampler(seed=0), pruner=NopPruner())
    trial = study.ask()
    overrides = walk_search_space(space, trial)
    assert set(overrides) == set(space)
    assert 0.01 <= overrides["training.learning_rate"] <= 0.3
    assert 3 <= overrides["training.max_depth"] <= 8
    assert overrides["training.colsample_bytree"] in {0.8, 1.0}


def test_walk_search_space_empty_returns_empty_dict() -> None:
    study = optuna.create_study(sampler=RandomSampler(seed=0))
    trial = study.ask()
    assert walk_search_space({}, trial) == {}


# ---------- build_objective end-to-end ----------


def test_build_objective_runs_to_completion(tune_cfg: RuxMLConfig) -> None:
    """K-fold CV-mean objective: 3 folds x 2 trials = 6 fits total."""
    objective = build_objective(tune_cfg)
    study = optuna.create_study(
        sampler=RandomSampler(seed=0),
        pruner=NopPruner(),
        direction="maximize",
    )
    study.optimize(objective, n_trials=2)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 2
    # Each trial recorded the 8-layer provenance set + metric.
    for t in completed:
        for layer in (
            "data_cfg_hash",
            "features_cfg_hash",
            "training_cfg_hash",
            "tuning_cfg_hash",
            "runs_cfg_hash",
            "registry_cfg_hash",
            "memory_cfg_hash",
            "cv_cfg_hash",
            "root_cfg_hash",
            "git_sha",
            "data_hash",
            "metric",
        ):
            assert layer in t.user_attrs


def test_build_objective_reports_per_fold(tune_cfg: RuxMLConfig) -> None:
    """``trial.report(fold_score, fold_idx)`` is called inside each trial."""
    objective = build_objective(tune_cfg)
    study = optuna.create_study(
        sampler=RandomSampler(seed=0),
        pruner=NopPruner(),
        direction="maximize",
    )
    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    # K=3 folds → 3 intermediate values keyed by fold index.
    assert set(trial.intermediate_values) == {0, 1, 2}


def test_build_objective_overrides_change_trial_cfg_hashes(tune_cfg: RuxMLConfig) -> None:
    """Different trial params → different per-trial cfg hashes (D17)."""
    objective = build_objective(tune_cfg)
    study = optuna.create_study(
        sampler=RandomSampler(seed=0),
        pruner=NopPruner(),
        direction="maximize",
    )
    study.optimize(objective, n_trials=2)
    a, b = study.trials
    # training overrides → training_cfg_hash should differ between trials
    # (the search space tunes training.* exclusively).
    assert a.user_attrs["training_cfg_hash"] != b.user_attrs["training_cfg_hash"]


def test_build_objective_extmem_gate_raises_for_incompatible_splitter(
    tune_cfg: RuxMLConfig,
) -> None:
    """ExtMem + non-time-series Splitter → NotImplementedError (PR-015 sub-decision C1)."""
    # Force the ingest path to ExtMemQuantileDMatrix by setting an absurdly small threshold.
    cfg = tune_cfg.model_copy(deep=True)
    cfg.data = DataConfig(
        source_path=tune_cfg.data.source_path,
        target_column=tune_cfg.data.target_column,
        gpu_in_memory_x_gb_max=1e-12,  # any X is "too big" → ExtMem path
    )
    # KFoldSplitter is not extmem_compatible.
    cfg.cv = KFoldCV(n_splits=3, shuffle=True)

    objective = build_objective(cfg)
    study = optuna.create_study(sampler=RandomSampler(seed=0), pruner=NopPruner())
    with pytest.raises(NotImplementedError, match="extmem-compatible"):
        study.optimize(objective, n_trials=1, catch=())


def test_build_objective_requires_source_path_and_target_column() -> None:
    cfg = RuxMLConfig()
    with pytest.raises(ValueError, match=r"source_path and data\.target_column"):
        build_objective(cfg)


def test_build_objective_stratified_kfold_requires_y_and_works(tune_cfg: RuxMLConfig) -> None:
    """StratifiedKFold path — y is passed through (PR-015 Splitter contract)."""
    cfg = tune_cfg.model_copy(deep=True)
    cfg.cv = StratifiedKFoldCV(n_splits=3, shuffle=True)
    objective = build_objective(cfg)
    study = optuna.create_study(
        sampler=RandomSampler(seed=0),
        pruner=NopPruner(),
        direction="maximize",
    )
    study.optimize(objective, n_trials=1)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 1
