"""PR-062 (program PR-082 R3-a; option A1): every fit path's booster runs with
``memory.omp_threads``, end to end through the verbs.

Read from the records (``booster_nthread``, the re-fit's ``nthread``), which come from
the booster's saved config — on the pre-PR code they record 0 (the OpenMP default), so
these fail on the assertion, not on an import.

- ``rux-ml train``: the sklearn wrapper (the default) and ``use_native``;
- ``rux-ml train`` of an ``[m9]`` regression (the per-day batch fit);
- ``rux-ml tune``: in-process and the subprocess trial child (the default);
- ``registry promote``'s in-memory re-fit (the ``[m9]`` re-fit: the batch fit above, and
  ``test_m9_promote_manifest_refit``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml._internal.seeds import make_seed_bag
from rux_ml.cli import app
from rux_ml.config import RuxMLConfig
from rux_ml.registry import promote as promote_module
from tests.cli.conftest import REPO, m9_argv
from tests.conftest import m9_gates_overrides, repo_oracle_toml


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A synthetic binary set + a base.toml for ``train`` and a 1-trial ``tune``
    (``test_train_subcommand`` / ``test_tune_subcommands``' shape). No ``[memory]``
    table: the budget is the default ``memory.omp_threads``."""
    rng = np.random.default_rng(0)
    n = 200
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    src = tmp_path / "synth.parquet"
    pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}).write_parquet(src)
    (tmp_path / "base.toml").write_text(
        f"""
[data]
source_path = "{src}"
target_column = "y"
cas_root = "{tmp_path}/cas"
manifests_root = "{tmp_path}/manifests"

[features]
spec = {{ numeric_columns = ["x1", "x2"], categorical_columns = [] }}

[training]
kind = "xgboost"
device = "cpu"
metric = "auc"
n_estimators = 8
max_depth = 3
learning_rate = 0.3

[tuning]
sampler = "tpe"
pruner = "wilcoxon"
n_trials = 1
n_startup_trials = 1
entropy = 42

[cv]
kind = "kfold"
n_splits = 3
shuffle = true

[runs]
storage_url = "sqlite:///{tmp_path}/studies/studies.db"
artifacts_root = "{tmp_path}/studies/artifacts"

[search_space."training.max_depth"]
type = "int"
low = 2
high = 4
"""
        + repo_oracle_toml()
    )
    return tmp_path


def _budget(workdir: Path) -> int:
    return RuxMLConfig.from_layers(workdir / "base.toml").memory.omp_threads


def _only_trial_attrs(tmp: Path) -> dict[str, Any]:
    storage = f"sqlite:///{tmp}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    (trial,) = optuna.load_study(study_name=summary.study_name, storage=storage).trials
    return trial.user_attrs


NATIVE = ("--set", "training.use_native=true")


@pytest.mark.parametrize("extra", [(), NATIVE], ids=["sklearn", "native"])
def test_a_default_train_records_the_budget(
    extra: tuple[str, ...], runner: CliRunner, workdir: Path
) -> None:
    """Test (4) (Verification 1): a default ``rux-ml train`` fit records
    ``booster_nthread == memory.omp_threads``."""
    argv = ["--config", str(workdir / "base.toml"), *extra, "train"]
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    attrs = _only_trial_attrs(workdir)
    assert attrs["omp_threads"] == _budget(workdir)
    assert attrs["booster_nthread"] == _budget(workdir)


@pytest.mark.parametrize("isolation", ["in_process", "subprocess"])
def test_a_tune_trial_records_the_budget(isolation: str, runner: CliRunner, workdir: Path) -> None:
    """The objective's per-fold trainer (``tuning/objective.py``), in-process and in the
    subprocess trial child — where xgboost also loads before ``pin_threads``."""
    argv = [
        "--config", str(workdir / "base.toml"), "--set", f'tuning.trial_isolation="{isolation}"',
        "tune", "start", f"budget_{isolation}", "--n-trials", "1",
    ]  # fmt: skip
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout
    assert _only_trial_attrs(workdir)["booster_nthread"] == _budget(workdir)


def test_the_in_memory_promote_refit_records_the_budget(workdir: Path) -> None:
    """``registry/promote.py``'s in-memory re-fit (the manifest's ``refit.nthread``)."""
    cfg = RuxMLConfig.from_layers(workdir / "base.toml")
    bag = make_seed_bag(master_entropy=42, trial_number=0)
    _, _, record = promote_module._refit(cfg, bag=bag, hashed_files=[])
    assert record.nthread == cfg.memory.omp_threads  # read from the booster's saved config


def test_a_default_m9_batch_fit_records_the_budget(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    """Verification 5: the ``[m9]`` per-day batch fit (``fit_batches``) at its default."""
    argv = m9_argv(tmp_path, c6_set, "m9_fill_frac", *m9_gates_overrides(signed_m9_gates))
    result = runner.invoke(app, [*argv, "train"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    budget = RuxMLConfig.from_layers(
        REPO / "configs" / "base.toml", problem="m9_fill_frac",
        problems_dir=REPO / "configs" / "problems",
    ).memory.omp_threads  # fmt: skip
    attrs = _only_trial_attrs(tmp_path)
    assert attrs["booster_nthread"] == budget
    assert attrs["booster_device"] == "cpu"
