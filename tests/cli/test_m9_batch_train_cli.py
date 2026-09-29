"""PR-054 (program PR-027 A1 / A3 / A11): ``rux-ml train`` on an [m9] problem, end to end.

Written against the CLI and the records only, so on the pre-PR code these fail on the
behaviour:

- **A1** the fit never collects the whole source (``materialize`` is not called);
- **A11** the trial records the booster's ``nthread`` and ``device`` (``pin_threads``'
  24 does not bound an XGBoost fit whose ``n_jobs`` is set);
- **A3** the label diagnostics, the split definition and the leakage audit a fit records
  equal the ones computed from the whole frame split in memory — the diagnostics are
  summarised by their own pass, never carried in a training frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import pytest
from typer.testing import CliRunner

from rux_ml._internal.seeds import make_seed_bag_from_hex
from rux_ml.cli import app
from rux_ml.cli import train as train_cli
from rux_ml.cli.train import _label_diagnostics  # pyright: ignore[reportPrivateUsage]
from rux_ml.config import RuxMLConfig
from rux_ml.data import load_parquet, make_splits, materialize, split_definition
from rux_ml.data.leakage import leakage_audit
from tests.cli.conftest import REPO, m9_argv
from tests.cli.test_m9_train_cli import _fold_meta  # pyright: ignore[reportPrivateUsage]
from tests.conftest import m9_gates_overrides

STUDIES = ["m9_regime_time_block", "m9_regime_row_random", "m9_regime_symbol_holdout"]


def _train(runner: CliRunner, tmp: Path, c6: Path, gates: dict[str, str], *extra: str) -> Any:
    argv = m9_argv(tmp, c6, "m9_fill_frac", *m9_gates_overrides(gates), *extra, "train")
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def _attrs(tmp: Path) -> dict[str, Any]:
    storage = f"sqlite:///{tmp}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    return optuna.load_study(study_name=summary.study_name, storage=storage).trials[0].user_attrs


def test_an_m9_train_never_materialises_the_whole_source(
    runner: CliRunner,
    c6_set: Path,
    tmp_path: Path,
    signed_m9_gates: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        msg = "the [m9] fit collected the whole source"
        raise AssertionError(msg)

    monkeypatch.setattr(train_cli, "materialize", refuse)
    result = _train(runner, tmp_path, c6_set, signed_m9_gates)
    assert "per-day batches" in result.stderr


def test_an_m9_trial_records_the_booster_threads_and_device(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    _train(runner, tmp_path, c6_set, signed_m9_gates, "--set", "training.model_kwargs.n_jobs=3")
    attrs = _attrs(tmp_path)
    assert attrs["booster_nthread"] == 3  # what the booster ran with, not OMP_NUM_THREADS
    assert attrs["booster_device"] == "cpu"
    assert _fold_meta(tmp_path, "m9_fill_frac")["booster"] == {"nthread": 3, "device": "cpu"}


@pytest.mark.parametrize("study", STUDIES)
def test_the_fit_records_equal_the_in_memory_records(
    study: str, runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    _train(runner, tmp_path, c6_set, signed_m9_gates, "--study", study)
    meta = _fold_meta(tmp_path, "m9_fill_frac")
    split_seed = make_seed_bag_from_hex(_attrs(tmp_path)["entropy_hex"]).split_seed
    cfg = RuxMLConfig.from_layers(
        REPO / "configs" / "base.toml",
        problem="m9_fill_frac",
        study=study,
        problems_dir=REPO / "configs" / "problems",
        studies_dir=REPO / "configs" / "studies",
        overrides={"data.source_path": str(c6_set / "fill")},
    )
    assert cfg.m9 is not None and cfg.data.source_path is not None
    df = materialize(load_parquet(cfg.data.source_path, oracle=cfg.data.oracle))
    parts = make_splits(cfg, df, seed=split_seed)
    assert meta["row_count"] == parts["train"].height
    assert meta["split_definition"] == split_definition(cfg, df, parts, seed=split_seed)
    window = cfg.m9.h_max_ms
    assert meta["leakage"] == leakage_audit(
        parts, time_column=cfg.data.time_column, group_column=cfg.data.group_column, window=window
    )
    want = _label_diagnostics(parts, cfg.m9.diagnostic_columns)
    for fold, cols in want.items():
        for c, stats in cols.items():
            for k, v in stats.items():
                g = meta["diagnostics"][fold][c][k]
                assert (g is None) == (v is None), (fold, c, k)
                if v is not None:
                    assert g == pytest.approx(v, rel=1e-12, abs=1e-12), (fold, c, k)
