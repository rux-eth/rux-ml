"""The three M9 problem configs (PR-041; program v0.3 D37 / D45 #3, ACCEPTANCE C9).

Each problem trains on exactly one label — the column the program's EV consumes —
and carries the other labels as diagnostics, never as features. The configs are
loaded through the real layered loader from the repo's ``configs/`` so the tests
lock what a study would actually run with.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import pytest
from xgboost import XGBRegressor

from rux_ml.config import RuxMLConfig, XGBoostTraining
from rux_ml.data import LabelAsFeatureError, check_feature_labels, check_label_quarantine
from rux_ml.training import make_trainer, task_for_metric
from tests.conftest import (
    m9_column_dtypes,
    m9_feature_columns,
    m9_nullable_features,
    m9_schema,
    m9_subtree,
)

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# problem -> (target, metric, diagnostics). PR-048: a problem's diagnostics are the other
# labels of ITS OWN subtree (program PR-024 A7: fill/ holds the post-only rows, walk/ the
# taker rows) — the materializer writes no walk label beside a fill row, nor the reverse.
_FILL_CURVES = {
    "y__fill_frac_h60000", "y__fill_frac_h600000", "y__fill_frac_h3600000",
    "y__markout_bp_h60000", "y__markout_bp_h600000", "y__markout_bp_h3600000",
    "y__capture_bp", "y__queue_ahead_at_ack", "y__through",
}  # fmt: skip
PROBLEMS: dict[str, tuple[str, str, set[str]]] = {
    "m9_fill_frac": ("y__fill_frac", "brier", _FILL_CURVES | {"y__markout_bp"}),
    "m9_markout_bp": ("y__markout_bp", "mae", _FILL_CURVES | {"y__fill_frac"}),
    "m9_walk_bp": (
        "y__walk_bp", "mae",
        {"y__walk_filled_frac", "y__half_spread_bp", "y__censored_beyond_book", "y__levels_walked"},
    ),
}  # fmt: skip
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


# ---------- program PR-024 A7 / A9 (operator-approved 2026-09-26) ----------

# problem -> the training set's per-target subtree (program PR-024 A7): ``fill/`` holds
# the post-only rows (y__fill_frac, y__markout_bp), ``walk/`` the taker rows (y__walk_bp).
SUBTREES: dict[str, str] = {
    "m9_fill_frac": "fill",
    "m9_markout_bp": "fill",
    "m9_walk_bp": "walk",
}


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_problem_reads_its_own_training_subtree(problem: str) -> None:
    cfg = _load(problem)
    assert cfg.data.source_path is not None
    assert cfg.data.source_path.name == SUBTREES[problem]
    # the three problems share one training root; only the subtree differs
    assert cfg.data.source_path.parent == _load("m9_fill_frac").data.source_path.parent  # type: ignore[union-attr]


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_problem_trains_and_scores_on_rows_with_its_target_only(problem: str) -> None:
    """PR-048 (program PR-024 A9 + B-4): the materializer WRITES the rows of an order
    never placed (no_book, below_one_lot) or resting nowhere (alo_expired) with NaN
    targets, the markout is null on an unfilled rung, and y__walk_bp is null where an
    IOC filled nothing — so every problem drops the rows its own target lacks."""
    cfg = _load(problem)
    assert cfg.m9 is not None
    assert cfg.m9.row_filter_non_null == [PROBLEMS[problem][0]]


# ---------- PR-048: every configured column exists in the problem's own subtree ----------

# The subtree's labels NOT summarised as diagnostics, each for a stated reason:
# y__outcome — the row filter keeps only rows whose target is present, which the schema
#   defines only for outcome ok: the summary would be a constant 0;
# y__ttf_first_ms / y__ttf_full_ms — "-1 when none": a sentinel the summary would average
#   in as a value (PR-041 counts missing values apart, never as values).
NOT_DIAGNOSTICS = {"y__outcome", "y__ttf_first_ms", "y__ttf_full_ms"}


def _subtree(cfg: RuxMLConfig) -> dict[str, Any]:
    assert cfg.data.source_path is not None
    return m9_subtree(cfg.data.source_path.name)


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_every_configured_column_is_in_the_problems_own_subtree_schema(problem: str) -> None:
    cfg = _load(problem)
    assert cfg.m9 is not None
    columns = set(_subtree(cfg)["columns"])
    configured = {
        "data.target_column": [cfg.data.target_column],
        "data.time_column": [cfg.data.time_column],
        "data.group_column": [cfg.data.group_column],
        "m9.diagnostic_columns": cfg.m9.diagnostic_columns,
        "m9.row_filter_non_null": cfg.m9.row_filter_non_null,
        "features.spec.numeric_columns": cfg.features.spec.numeric_columns,
        "features.spec.categorical_columns": cfg.features.spec.categorical_columns,
    }
    absent = {k: [c for c in v if c not in columns] for k, v in configured.items()}
    assert not {k: v for k, v in absent.items() if v}, absent


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_diagnostics_are_the_subtrees_other_labels(problem: str) -> None:
    cfg = _load(problem)
    assert cfg.m9 is not None
    labels = {c["name"] for c in _subtree(cfg)["y"]}
    expected = labels - {cfg.data.target_column} - NOT_DIAGNOSTICS
    assert set(cfg.m9.diagnostic_columns) == expected


def _ev_targets(subtree: dict[str, Any]) -> set[str]:
    return {c["name"] for c in subtree["y"] if "the EV's target" in c["definition"]}


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_target_is_the_schemas_unsuffixed_ev_target(problem: str) -> None:
    """The EV consumes the unsuffixed name (D45 #3); the horizon curve carries ``_h<ms>``."""
    cfg = _load(problem)
    assert cfg.data.target_column == PROBLEMS[problem][0]
    assert cfg.data.target_column in _ev_targets(_subtree(cfg))
    assert not re.search(r"_h\d+$", cfg.data.target_column)


def test_each_ev_target_of_the_schema_is_one_problems_target() -> None:
    ev = _ev_targets(m9_subtree("fill")) | _ev_targets(m9_subtree("walk"))
    targets = [_load(p).data.target_column for p in sorted(PROBLEMS)]
    assert sorted(targets) == sorted(ev) == ["y__fill_frac", "y__markout_bp", "y__walk_bp"]


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_categorical_features_are_strings_in_the_schema(problem: str) -> None:
    """Polars casts only a string column to Categorical: ``side`` is int8 (+1 / -1) in
    the materialized set, so it must be numeric — ``cast(pl.Categorical)`` raises on it."""
    cfg = _load(problem)
    assert cfg.data.source_path is not None
    dtypes = m9_column_dtypes(cfg.data.source_path.name)
    assert {c: dtypes[c] for c in cfg.features.spec.categorical_columns} == {
        c: "string" for c in cfg.features.spec.categorical_columns
    }


@pytest.mark.skipif(
    not os.environ.get("RUXML_M9_SCHEMA_PATH"), reason="RUXML_M9_SCHEMA_PATH not set"
)
def test_vendored_schema_matches_the_live_harness_schema() -> None:
    """The copy under tests/fixtures/ against the harness's live file (drift check)."""
    live = m9_schema(Path(os.environ["RUXML_M9_SCHEMA_PATH"]))
    vendored = m9_schema()
    assert vendored["targets"] == live["targets"]
    assert vendored["key_dtypes"] == live["key_dtypes"]


# ---------- PR-049: the features = the subtree's feat__ columns + the order attributes ----------

# The order's own attributes, carried since PR-041 (PR-048: ``side`` is int8, so numeric).
ORDER_NUMERIC = ["p_bp", "q_usd", "h_ms", "side"]
ORDER_CATEGORICAL = ["kind"]
# spec/m9_training_schema.json v2: 28 feat__ columns in fill/, 27 in walk/ —
# feat__queue_ahead_usd (the queue at the order's own price) exists only for a resting order.
FEAT_COUNTS = {"m9_fill_frac": 28, "m9_markout_bp": 28, "m9_walk_bp": 27}
# The schema's feature definitions that say the value can be empty ("null when ...").
_NULLABLE_BOTH = {
    "feat__near_walk_bp_q1000", "feat__near_walk_bp_q3000", "feat__ofi_60s", "feat__ofi_300s",
    "feat__rv_bp_300s", "feat__rv_bp_3600s",
}  # fmt: skip
NULLABLE = {"fill": _NULLABLE_BOTH | {"feat__queue_ahead_usd"}, "walk": _NULLABLE_BOTH}
_NUMERIC_DTYPES = {"int8", "int16", "int32", "int64", "float32", "float64"}


def _subtree_name(cfg: RuxMLConfig) -> str:
    assert cfg.data.source_path is not None
    return cfg.data.source_path.name


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_features_are_the_subtrees_feat_columns_plus_the_order_attributes(problem: str) -> None:
    """Program D38: the model's features are the ``feat__`` columns (C10: the list comes
    from the schema file). Each problem file names them explicitly, in the schema's order,
    so a schema change fails here until the config follows."""
    cfg = _load(problem)
    subtree = _subtree_name(cfg)
    feats = m9_feature_columns(subtree)
    assert feats == [c["name"] for c in m9_subtree(subtree)["feat"]]  # columns and entries agree
    assert len(feats) == FEAT_COUNTS[problem]
    assert cfg.features.spec.numeric_columns == [*ORDER_NUMERIC, *feats]
    assert cfg.features.spec.categorical_columns == ORDER_CATEGORICAL


def test_queue_ahead_is_a_feature_of_the_resting_order_only() -> None:
    for problem, present in [
        ("m9_fill_frac", True),
        ("m9_markout_bp", True),
        ("m9_walk_bp", False),
    ]:
        cfg = _load(problem)
        assert ("feat__queue_ahead_usd" in cfg.features.spec.numeric_columns) is present, problem


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_numeric_features_are_numeric_in_the_schema(problem: str) -> None:
    """Nothing in the list is misdeclared: a numeric feature has a numeric schema dtype
    (PR-048 found ``side``, int8, declared categorical — a Polars cast error)."""
    cfg = _load(problem)
    dtypes = m9_column_dtypes(_subtree_name(cfg))
    misdeclared = {
        c: dtypes[c] for c in cfg.features.spec.numeric_columns if dtypes[c] not in _NUMERIC_DTYPES
    }
    assert not misdeclared
    assert not set(cfg.features.spec.numeric_columns) & set(cfg.features.spec.categorical_columns)


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_the_feature_list_passes_the_label_quarantine_and_the_refusal_still_bites(
    problem: str,
) -> None:
    """PR-043: no feat__ column is a label; a y__ column added to the same list is refused."""
    cfg = _load(problem)
    check_feature_labels(cfg)
    assert cfg.m9 is not None
    spec = cfg.features.spec
    for label in (cfg.data.target_column, cfg.m9.diagnostic_columns[0], "Y__anything"):
        with pytest.raises(LabelAsFeatureError):
            check_label_quarantine(
                [*spec.numeric_columns, label, *spec.categorical_columns],
                cfg.data.oracle,
                diagnostics=cfg.m9.diagnostic_columns,
            )


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_nullable_features_are_listed_and_fed_to_a_nan_native_family(problem: str) -> None:
    """The schema's possibly-empty features (a thin book, no l2 transition, beyond the
    deepest displayed level) are features too: the family is XGBoost, whose ``missing``
    is NaN — a null arrives as NaN (Polars -> pandas) and takes the learned default branch.
    The CLI test ``test_the_fit_sees_the_feat_columns_with_nulls_as_missing`` drives it."""
    cfg = _load(problem)
    subtree = _subtree_name(cfg)
    nullable = set(m9_nullable_features(subtree))
    assert nullable == NULLABLE[subtree]
    assert nullable <= set(cfg.features.spec.numeric_columns)
    assert isinstance(cfg.training, XGBoostTraining)
    assert "missing" not in cfg.training.model_kwargs
    assert math.isnan(make_trainer(cfg.training).get_params()["missing"])


# ---------- PR-058 (program PR-027 A7): the honesty forms each problem carries ----------


@pytest.mark.parametrize(
    ("problem", "form", "buckets"),
    [
        ("m9_fill_frac", None, ["p_bp", "q_usd", "h_ms"]),
        ("m9_markout_bp", "clipped_at_zero", []),
        ("m9_walk_bp", "signed", []),
    ],
)
def test_repo_problems_carry_the_a7_honesty_forms(
    problem: str, form: str | None, buckets: list[str]
) -> None:
    """Program PR-027 A7 (:690-691, operator-approved 2026-09-28): fill is gated on Brier with
    a signed calibration bias per (p, Q, h) bucket reported; markout is tested as the EV
    deducts it, max(0, m); the walk keeps PR-044's signed form (A7 names no clip for it)."""
    cfg = _load(problem)
    assert cfg.m9 is not None
    if form is None:
        assert cfg.m9.signed_error_honesty is False
    else:
        assert cfg.m9.signed_error_honesty is True and cfg.m9.honesty_verdict_form == form
    assert cfg.m9.calibration_bias is bool(buckets)
    assert cfg.m9.calibration_bucket_columns == buckets
    # every bucket column is a row key, so the fit's own export re-derives the record
    assert set(buckets) <= set(cfg.m9.row_key_columns)


def test_m9_config_refuses_a_dead_or_unknown_a7_setting() -> None:
    from pydantic import ValidationError  # noqa: PLC0415

    from rux_ml.config.m9 import M9Config  # noqa: PLC0415

    with pytest.raises(ValidationError, match="calibration_bias"):
        M9Config(calibration_bucket_columns=["p_bp"])
    with pytest.raises(ValidationError, match="signed_error_honesty"):
        M9Config(honesty_verdict_form="clipped_at_zero")
    with pytest.raises(ValidationError, match="distinct"):
        M9Config(calibration_bias=True, calibration_bucket_columns=["p_bp", "p_bp"])
    with pytest.raises(ValidationError, match="honesty_verdict_form"):
        M9Config(honesty_verdict_form="clipped")  # type: ignore[arg-type]
    assert M9Config().honesty_verdict_form == "signed" and M9Config().calibration_bias is False
