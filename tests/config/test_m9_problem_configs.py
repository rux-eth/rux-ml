"""The three M9 problem configs (PR-041; program v0.3 D37 / D45 #3, ACCEPTANCE C9).

Each problem trains on exactly one label — the column the program's EV consumes —
and carries the other labels as diagnostics, never as features. The configs are
loaded through the real layered loader from the repo's ``configs/`` so the tests
lock what a study would actually run with.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from xgboost import XGBRegressor

from rux_ml.config import RuxMLConfig, XGBoostTraining
from rux_ml.training import make_trainer, task_for_metric

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# problem -> (target, metric, the other two labels as diagnostics)
PROBLEMS: dict[str, tuple[str, str, set[str]]] = {
    "m9_fill_frac": ("y__fill_frac", "brier", {"y__markout_bp", "y__walk_bp"}),
    "m9_markout_bp": ("y__markout_bp", "mae", {"y__fill_frac", "y__walk_bp"}),
    "m9_walk_bp": ("y__walk_bp", "mae", {"y__fill_frac", "y__markout_bp"}),
}
# The training table's keys (program ACCEPTANCE C6): never features.
KEYS = {"stamp_ms", "coin", "side", "kind", "p_bp", "q_usd", "h_ms"}
# The order's own attributes are legitimate inputs of P(fill | p, Q, h) (D37 #1).
ORDER_ATTRIBUTES = {"p_bp", "q_usd", "h_ms", "side", "kind"}


def _load(problem: str) -> RuxMLConfig:
    return RuxMLConfig.from_layers(
        REPO_CONFIGS / "base.toml", problem=problem, problems_dir=REPO_CONFIGS / "problems"
    )


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_problem_targets_one_label_and_lists_the_others_as_diagnostics(problem: str) -> None:
    target, metric, diagnostics = PROBLEMS[problem]
    cfg = _load(problem)
    assert cfg.data.target_column == target
    assert cfg.training.metric == metric
    assert cfg.m9 is not None
    assert set(cfg.m9.diagnostic_columns) == diagnostics
    assert target not in cfg.m9.diagnostic_columns


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_problem_features_carry_no_label_and_no_key_beyond_the_order_attributes(
    problem: str,
) -> None:
    cfg = _load(problem)
    features = {*cfg.features.spec.numeric_columns, *cfg.features.spec.categorical_columns}
    assert not [c for c in features if c.startswith("y__")]
    assert not [c for c in features if c.startswith("oracle__")]
    assert features & KEYS == ORDER_ATTRIBUTES
    assert not (features & set(cfg.m9.diagnostic_columns if cfg.m9 else []))


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_problem_one_off_split_is_time_ordered_on_the_stamp(problem: str) -> None:
    cfg = _load(problem)
    assert cfg.data.split_kind == "time_ordered"
    assert cfg.data.time_column == "stamp_ms"


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_problem_is_a_default_config_regressor(problem: str) -> None:
    """C9: 'default-config GBT per target' — no hyperparameter is set in the problem file."""
    cfg = _load(problem)
    assert isinstance(cfg.training, XGBoostTraining)
    assert task_for_metric(cfg.training.metric) == "regression"
    assert isinstance(make_trainer(cfg.training), XGBRegressor)
    defaults = XGBoostTraining(kind="xgboost")
    for name in ("learning_rate", "max_depth", "n_estimators", "subsample", "colsample_bytree"):
        assert getattr(cfg.training, name) == getattr(defaults, name), name


def test_fill_fraction_is_fitted_as_a_probability() -> None:
    """A fraction in [0, 1] scored by Brier trains under XGBoost's logistic regression."""
    cfg = _load("m9_fill_frac")
    assert isinstance(cfg.training, XGBoostTraining)
    assert cfg.training.model_kwargs.get("objective") == "reg:logistic"
    trainer = make_trainer(cfg.training)
    assert isinstance(trainer, XGBRegressor)
    assert trainer.get_params()["objective"] == "reg:logistic"


@pytest.mark.parametrize("problem", ["m9_markout_bp", "m9_walk_bp"])
def test_cost_problems_use_no_objective_override(problem: str) -> None:
    cfg = _load(problem)
    assert isinstance(cfg.training, XGBoostTraining)
    assert "objective" not in cfg.training.model_kwargs
