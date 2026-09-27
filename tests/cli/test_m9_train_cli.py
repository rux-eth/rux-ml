"""PR-044: an M9 fit reports its held-out score and the signed-error honesty verdict.

Drives the repo's real M9 problem configs through the real CLI on a synthetic set
with the program's C6 layout, ``[m9.gates]`` pointed at a gates file signed here
with a throwaway key (tests/conftest.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import polars as pl
import pytest
from optuna.artifacts import download_artifact
from typer.testing import CliRunner
from xgboost import XGBRegressor

from rux_ml.cli import app
from rux_ml.config import RuxMLConfig
from rux_ml.runs import list_trial_artifacts, make_artifact_store
from tests.cli.conftest import REPO, m9_argv, problem_subtree
from tests.conftest import (
    M9_GATE_VALUES,
    m9_feature_columns,
    m9_gates_overrides,
    m9_nullable_features,
)


def _fold_meta(tmp: Path, problem: str) -> dict[str, Any]:
    storage = f"sqlite:///{tmp}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    metas = list_trial_artifacts(storage, summary.study_name, 0)
    fold_meta_id = next(m.artifact_id for m in metas if m.filename == "fold_meta.json")
    cfg = RuxMLConfig.from_layers(
        REPO / "configs" / "base.toml",
        problem=problem,
        problems_dir=REPO / "configs" / "problems",
        overrides={"runs.artifacts_root": str(tmp / "studies" / "artifacts")},
    )
    store = make_artifact_store(cfg, study_name=summary.study_name)
    out = tmp / "fold_meta.json"
    download_artifact(artifact_store=store, artifact_id=fold_meta_id, file_path=str(out))
    return json.loads(out.read_text())[0]


def test_walk_fit_reports_the_honesty_verdict_at_the_signed_thresholds(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    argv = m9_argv(tmp_path, c6_set, "m9_walk_bp", *m9_gates_overrides(signed_m9_gates), "train")
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    oos = _fold_meta(tmp_path, "m9_walk_bp")["oos"]
    sha = hashlib.sha256(Path(signed_m9_gates["path"]).read_bytes()).hexdigest()
    assert oos["partition"] == "test" and oos["gates_sha256"] == sha
    assert oos["metric"] == "mae" and oos["n_scored"] == oos["n_rows"] > 0
    h = oos["honesty"]
    assert h["n"] == oos["n_rows"]
    assert h["no_underdeduct_frac_min"] == M9_GATE_VALUES["m9_honesty_no_underdeduct_frac_min"]
    assert h["overdeduct_max_rel"] == M9_GATE_VALUES["m9_honesty_overdeduct_max_rel"]
    assert h["honest"] == (h["passes_no_underdeduct"] and h["passes_overdeduct"])
    assert abs(h["mae_bp"] - oos["score"]) < 1e-9  # the default metric, beside it


def test_fill_fit_reports_brier_and_no_honesty_test(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    argv = m9_argv(tmp_path, c6_set, "m9_fill_frac", *m9_gates_overrides(signed_m9_gates), "train")
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    oos = _fold_meta(tmp_path, "m9_fill_frac")["oos"]
    assert oos["metric"] == "brier" and 0.0 <= oos["score"] <= 1.0
    assert "honesty" not in oos and "gates_sha256" in oos


def test_unsigned_keys_refuse_the_run_with_exit_2_and_no_trial(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    gates = Path(signed_m9_gates["path"])
    gates.write_text(gates.read_text() + "# edited after signing\n")
    argv = m9_argv(tmp_path, c6_set, "m9_walk_bp", *m9_gates_overrides(signed_m9_gates), "train")
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "m9 gates" in result.output
    assert not (tmp_path / "studies" / "studies.db").exists() or not optuna.get_all_study_summaries(
        storage=f"sqlite:///{tmp_path}/studies/studies.db"
    )


# ---------- PR-048: a NaN / null target never reaches XGBoost, on every problem ----------

TARGETS = {
    "m9_fill_frac": "y__fill_frac",
    "m9_markout_bp": "y__markout_bp",
    "m9_walk_bp": "y__walk_bp",
}


@pytest.mark.parametrize("problem", sorted(TARGETS))
def test_no_missing_target_reaches_the_xgboost_fit(
    problem: str,
    runner: CliRunner,
    c6_set: Path,
    tmp_path: Path,
    signed_m9_gates: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Program PR-024 B-4 writes the no_book / below_one_lot / alo_expired rows with NaN
    targets, a null markout on an unfilled rung (A9) and a null y__walk_bp where an IOC
    filled nothing. Each problem's row predicate drops the rows its target lacks before
    the split: neither the fitted nor the early-stopping target XGBoost receives holds a
    missing value, every held-out row is scored, and the drop is recorded apart."""
    target = TARGETS[problem]
    subtree = c6_set / problem_subtree(problem)
    missing = (
        pl.scan_parquet(subtree)
        .select((pl.col(target).is_null() | pl.col(target).is_nan()).sum())
        .collect()
        .item()
    )
    assert missing > 0  # the fixture carries rows without the target

    seen: list[np.ndarray] = []
    original_fit = XGBRegressor.fit

    def spy(self: XGBRegressor, x: object, y: object, **kw: object) -> object:
        seen.append(np.asarray(y, dtype=float))
        for _, y_eval in kw.get("eval_set") or []:  # type: ignore[union-attr]
            seen.append(np.asarray(y_eval, dtype=float))
        return original_fit(self, x, y, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(XGBRegressor, "fit", spy)
    argv = m9_argv(tmp_path, c6_set, problem, *m9_gates_overrides(signed_m9_gates), "train")
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert len(seen) == 2  # the fitted target and the eval-set target
    assert not any(np.isnan(y).any() for y in seen)
    meta = _fold_meta(tmp_path, problem)
    assert meta["split_definition"]["row_filter_non_null"] == [target]
    assert meta["split_definition"]["row_filter_dropped"] == missing
    oos = meta["oos"]
    assert oos["n_scored"] == oos["n_rows"] > 0


# ---------- PR-049: every feat__ column reaches the fit, a null as XGBoost's missing ----------


@pytest.mark.parametrize("problem", sorted(TARGETS))
def test_the_fit_sees_the_feat_columns_with_nulls_as_missing(
    problem: str,
    runner: CliRunner,
    c6_set: Path,
    tmp_path: Path,
    signed_m9_gates: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Program D38 / C10: the model's features are the subtree's ``feat__`` columns (plus
    the order's own attributes). On a set whose possibly-empty features carry nulls, the
    frame XGBoost fits (and early-stops on) holds exactly those columns, the numeric ones
    as numbers, and each null arrives as NaN — XGBoost's ``missing`` — never refused,
    imputed or dropped."""
    subtree = problem_subtree(problem)
    feats = m9_feature_columns(subtree)
    nullable = m9_nullable_features(subtree)
    nulls = (
        pl.scan_parquet(c6_set / subtree)
        .select([pl.col(c).null_count() for c in nullable])
        .collect()
        .row(0)
    )
    assert nullable
    assert all(n > 0 for n in nulls)  # the fixture writes the schema's empties

    seen: list[pd.DataFrame] = []
    original_fit = XGBRegressor.fit

    def spy(self: XGBRegressor, x: pd.DataFrame, y: object, **kw: object) -> object:
        seen.append(x)
        seen.extend(x_eval for x_eval, _ in kw.get("eval_set") or [])  # type: ignore[union-attr]
        return original_fit(self, x, y, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(XGBRegressor, "fit", spy)
    argv = m9_argv(tmp_path, c6_set, problem, *m9_gates_overrides(signed_m9_gates), "train")
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert len(seen) == 2  # the fitted frame and the eval-set frame
    numeric = ["p_bp", "q_usd", "h_ms", "side", *feats]
    for x in seen:
        assert list(x.columns) == [*numeric, "kind"]
        assert all(pd.api.types.is_numeric_dtype(x[c]) for c in numeric)
        assert isinstance(x["kind"].dtype, pd.CategoricalDtype)
        assert all(x[c].isna().any() for c in nullable), "a null did not reach XGBoost as NaN"
        assert not x[[c for c in numeric if c not in nullable]].isna().any().any()
    assert np.isnan(XGBRegressor().get_params()["missing"])
    oos = _fold_meta(tmp_path, problem)["oos"]
    assert oos["n_scored"] == oos["n_rows"] > 0  # held-out rows with empty features are scored


# ---------- PR-045: the three regimes, audited on every fit ----------


def _fit(
    runner: CliRunner, tmp: Path, c6: Path, gates: dict[str, str], study: str
) -> dict[str, Any]:
    argv = m9_argv(tmp, c6, "m9_walk_bp", *m9_gates_overrides(gates), "--study", study, "train")
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return _fold_meta(tmp, "m9_walk_bp")


def test_row_random_regime_records_leakage_on_both_axes(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    meta = _fit(runner, tmp_path, c6_set, signed_m9_gates, "m9_regime_row_random")
    leak = meta["leakage"]
    assert meta["split_definition"]["kind"] == "random"
    assert leak["window"] == 28_801_000  # [m9] h_max_ms = the label's reach (program PR-024 A3)
    assert leak["stamp_violations"]["test_vs_fitted"] > 0
    assert leak["group_overlap"]["train&test"] > 0


def test_time_block_regime_has_no_fitted_stamp_within_h_max(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    meta = _fit(runner, tmp_path, c6_set, signed_m9_gates, "m9_regime_time_block")
    assert meta["leakage"]["stamp_violations"] == {"val_vs_train": 0, "test_vs_fitted": 0}
    assert meta["split_definition"]["purged_rows"] > 0


def test_symbol_holdout_regime_has_no_holdout_coin_in_train(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    meta = _fit(runner, tmp_path, c6_set, signed_m9_gates, "m9_regime_symbol_holdout")
    assert meta["leakage"]["group_overlap"] == {"train&val": 0, "train&test": 0, "val&test": 0}
    groups = meta["split_definition"]["groups"]
    assert groups == {"train": sorted(groups["train"]), "val": ["SUI"], "test": ["ADA", "AVAX"]}


# ---------- PR-046: the learning-curve report over four prefix fits ----------


def test_learning_curve_report_over_the_four_prefix_fits(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    head = m9_argv(tmp_path, c6_set, "m9_walk_bp", *m9_gates_overrides(signed_m9_gates))
    for frac in (0.25, 0.5, 0.75, None):  # the full window runs without the knob
        extra = [] if frac is None else ["--set", f"data.train_prefix_frac={frac}"]
        result = runner.invoke(app, [*head, *extra, "train"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    studies = [s.study_name for s in optuna.get_all_study_summaries(storage=storage)]
    argv = [*head, "runs", "learning-curve", "--output", str(tmp_path / "lc.json")]
    for name in studies:
        argv += ["--study", name]
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "lc.json").read_text())
    assert [p["train_prefix_frac"] for p in report["curve"]] == [0.25, 0.5, 0.75, 1.0]
    assert report["tolerance_rel"] == M9_GATE_VALUES["m9_learning_curve_tolerance_rel"]
    assert (
        report["gates_sha256"]
        == hashlib.sha256(Path(signed_m9_gates["path"]).read_bytes()).hexdigest()
    )
    prev, last = report["curve"][2]["value"], report["curve"][3]["value"]
    assert report["history_limited"] == ((prev - last) / abs(prev) > report["tolerance_rel"])
    assert report["regime"] == "time_ordered" and report["target_column"] == "y__walk_bp"


def test_learning_curve_refuses_an_incomplete_curve(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    head = m9_argv(tmp_path, c6_set, "m9_walk_bp", *m9_gates_overrides(signed_m9_gates))
    result = runner.invoke(app, [*head, "train"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    argv = [*head, "runs", "learning-curve", "--study", summary.study_name]
    result = runner.invoke(app, [*argv, "--output", str(tmp_path / "lc.json")])
    assert result.exit_code == 2, result.output
    assert "learning curve" in result.output
    assert not (tmp_path / "lc.json").exists()
