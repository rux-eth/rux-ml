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

import optuna
import polars as pl
from optuna.artifacts import download_artifact
from typer.testing import CliRunner

from rux_ml.cli import app
from rux_ml.config import RuxMLConfig
from rux_ml.runs import list_trial_artifacts, make_artifact_store
from tests.cli.conftest import REPO, m9_argv
from tests.conftest import M9_GATE_VALUES, m9_gates_overrides


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


def test_markout_fit_trains_and_scores_on_filled_rows_only(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    """Program PR-024 A9: y__markout_bp is null on an unfilled rung. The markout
    problem's row predicate drops those rows before the split, so every fitted and
    held-out row has a target (n_scored == n_rows) and the drop is recorded apart."""
    nulls = pl.scan_parquet(c6_set).select(pl.col("y__markout_bp").null_count()).collect().item()
    assert nulls > 0  # the fixture carries unfilled rungs
    argv = m9_argv(
        tmp_path, c6_set, "m9_markout_bp", *m9_gates_overrides(signed_m9_gates), "train"
    )
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    meta = _fold_meta(tmp_path, "m9_markout_bp")
    assert meta["split_definition"]["row_filter_non_null"] == ["y__markout_bp"]
    assert meta["split_definition"]["row_filter_dropped"] == nulls
    oos = meta["oos"]
    assert oos["metric"] == "mae" and oos["n_scored"] == oos["n_rows"] > 0
