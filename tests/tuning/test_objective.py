"""Tests for ``walk_search_space`` + ``build_objective``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import polars as pl
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
    TimeSeriesSplitCV,
    TuningConfig,
    XGBoostTraining,
)
from rux_ml.config.features import FeaturesConfig, FeaturesSpec
from rux_ml.data import load_parquet, materialize
from rux_ml.tuning import build_objective, walk_search_space
from rux_ml.tuning.objective import _carve_substrate
from tests.conftest import repo_oracle_cfg

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


def test_build_objective_uploads_diagnostics_post_fold_scores(
    tune_cfg: RuxMLConfig,
) -> None:
    """PR-034: each completed trial uploads metrics.json + fold_meta.json.

    Asserts Optuna's auto-persisted ``trial.system_attrs["artifacts:..."]``
    entries are present (Probe 3 binding-at-creation), the per-study
    base_path exists (Q5 Optuna FAQ layout), and a round-trip through
    ``get_all_artifact_meta`` reads back both files.
    """
    objective = build_objective(tune_cfg)
    storage_path = tune_cfg.runs.artifacts_root.parent / "studies.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")
    study = optuna.create_study(
        study_name="upload_test",
        storage=storage,
        sampler=RandomSampler(seed=0),
        pruner=NopPruner(),
        direction="maximize",
    )
    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    # Probe 3 binding-at-creation: 2 artifacts (metrics.json + fold_meta.json) in system_attrs.
    artifact_keys = [k for k in trial.system_attrs if k.startswith("artifacts:")]
    assert len(artifact_keys) == 2
    # Per-study layout per Optuna 4.8 FAQ.
    expected_dir = tune_cfg.runs.artifacts_root / "upload_test"
    assert expected_dir.exists()
    assert len(list(expected_dir.iterdir())) == 2


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
        oracle=tune_cfg.data.oracle,
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


# ---------- PR-031: substrate carve ----------


def _make_substrate_cfg(
    src: Path,
    *,
    split_kind: str,
    time_column: str | None,
    cv_kind: str,
) -> RuxMLConfig:
    """Tiny RuxMLConfig with the requested split_kind + cv. PR-031 substrate carve test scaffold."""
    data_kwargs: dict[str, object] = {
        "source_path": src,
        "target_column": "y",
        "oracle": repo_oracle_cfg(),
    }
    if split_kind == "time_ordered":
        data_kwargs["split_kind"] = "time_ordered"
        assert time_column is not None
        data_kwargs["time_column"] = time_column
    cv: KFoldCV | TimeSeriesSplitCV = (
        TimeSeriesSplitCV(n_splits=3, time_column=time_column)
        if cv_kind == "time_series"
        else KFoldCV(n_splits=3, shuffle=True)
    )
    return RuxMLConfig(
        data=DataConfig(**data_kwargs),  # pyright: ignore[reportArgumentType]
        features=FeaturesConfig(
            spec=FeaturesSpec(numeric_columns=["x1"], categorical_columns=[]),
        ),
        training=XGBoostTraining(
            device="cpu",
            metric="auc",
            n_estimators=4,
            max_depth=2,
            learning_rate=0.3,
            early_stopping_rounds=None,
        ),
        tuning=TuningConfig(entropy=42),
        cv=cv,
    )


def test_carve_substrate_excludes_test_fold_for_time_ordered(tmp_path: Path) -> None:
    """PR-031: substrate = train+val; test fold (last 15% for time_ordered) excluded.

    Default ``data.split_ratios = {train: 0.7, val: 0.15, test: 0.15}`` →
    substrate is first 85 of 100 rows; ``splits["test"]`` (last 15 rows) is
    truly held out from HP search.
    """
    n = 100
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "ts": list(range(n)),
            "x1": rng.normal(size=n).tolist(),
            "y": rng.integers(0, 2, size=n).tolist(),
        }
    )
    src = tmp_path / "synth.parquet"
    df.write_parquet(src)

    cfg = _make_substrate_cfg(
        src, split_kind="time_ordered", time_column="ts", cv_kind="time_series"
    )
    df_full = materialize(load_parquet(src, oracle=repo_oracle_cfg()))
    x_sub, y_sub = _carve_substrate(cfg, df_full, "y")

    assert x_sub.height == 85, "default 0.7+0.15 = 85% substrate"
    assert y_sub.len() == 85
    # ts column preserved (not the target); max ts in substrate is row 84
    # (rows 85..99 form the held-out test fold).
    assert x_sub["ts"].max() == 84


def test_carve_substrate_random_is_deterministic_across_invocations(tmp_path: Path) -> None:
    """PR-031: random-split substrate is reproducible per study identity.

    Study-level seed (``make_seed_bag(trial_number=0).split_seed``) makes the
    substrate identical across repeated invocations — the contract that lets
    every trial in a study CV over the same rows.
    """
    n = 100
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "x1": rng.normal(size=n).tolist(),
            "y": rng.integers(0, 2, size=n).tolist(),
        }
    )
    src = tmp_path / "synth.parquet"
    df.write_parquet(src)

    cfg = _make_substrate_cfg(src, split_kind="random", time_column=None, cv_kind="kfold")
    df_full = materialize(load_parquet(src, oracle=repo_oracle_cfg()))
    x_sub_a, y_sub_a = _carve_substrate(cfg, df_full, "y")
    x_sub_b, y_sub_b = _carve_substrate(cfg, df_full, "y")

    assert x_sub_a.equals(x_sub_b)
    assert y_sub_a.equals(y_sub_b)
    assert x_sub_a.height == 85
