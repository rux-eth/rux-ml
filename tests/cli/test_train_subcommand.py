"""End-to-end CLI tests for ``rux-ml train``.

Tiny synthetic Parquet → split (random OR time_ordered) → features → XGBoost
→ score, recorded as a 1-trial Optuna study in SQLite (per D7).

PR-024 adds a ``time_ordered`` dispatch path via ``cfg.data.split_kind``;
the CLI integration test below exercises it end-to-end through
:func:`rux_ml.data.splits.make_splits` and the cross-field validator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest
from optuna.artifacts import download_artifact
from typer.testing import CliRunner
from xgboost import XGBClassifier

from rux_ml.cli import app
from rux_ml.cli.train import _label_diagnostics  # pyright: ignore[reportPrivateUsage]
from rux_ml.config import RuxMLConfig
from rux_ml.runs import list_trial_artifacts, make_artifact_store
from tests.conftest import repo_oracle_toml


@pytest.fixture
def train_workdir(tmp_path: Path) -> Path:
    """Workdir with a synthetic Parquet + base.toml pointing every path under tmp_path."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Linearly-separable-ish binary target with noise (a baseline XGBoost should do >> 0.5 AUC).
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    df = pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()})
    src = tmp_path / "synth.parquet"
    df.write_parquet(src)

    storage_path = tmp_path / "studies" / "studies.db"
    config = tmp_path / "base.toml"
    config.write_text(
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
n_estimators = 16
max_depth = 3
learning_rate = 0.3
# `early_stopping_rounds` omitted — TOML has no null; keeping the Pydantic
# default of 50 is fine since n_estimators=16 trains to completion anyway.

[runs]
storage_url = "sqlite:///{storage_path}"
artifacts_root = "{tmp_path}/studies/artifacts"
"""
        + repo_oracle_toml()
    )
    return tmp_path


def _argv(workdir: Path, *args: str) -> list[str]:
    return ["--config", str(workdir / "base.toml"), *args]


def test_train_runs_end_to_end_cpu(runner: CliRunner, train_workdir: Path) -> None:
    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "score (auc):" in result.stdout
    # AUC must be a valid float in [0, 1]; on this synthetic separable set we
    # expect well above chance, but the band is loose to keep CI stable.
    score_line = next(line for line in result.stdout.splitlines() if "score (auc):" in line)
    score = float(score_line.split(":")[-1].strip())
    assert 0.0 <= score <= 1.0
    assert score > 0.6, f"AUC suspiciously low for separable data: {score}"


def test_train_records_user_attrs_in_sqlite(runner: CliRunner, train_workdir: Path) -> None:
    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout

    storage = f"sqlite:///{train_workdir}/studies/studies.db"
    summaries = optuna.get_all_study_summaries(storage=storage)
    assert len(summaries) == 1
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    assert len(study.trials) == 1
    attrs = study.trials[0].user_attrs

    # Provenance subset must all be present (cv_cfg_hash added by PR-015,
    # peak_rss_mb added + tightened to required by PR-011).
    expected = {
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
        "data_bytes_hash",
        "data_logical_hash",
        "metric",
        "peak_rss_mb",
    }
    assert expected.issubset(attrs.keys()), expected - attrs.keys()
    assert attrs["metric"] == "auc"
    assert attrs["peak_rss_mb"] > 0  # watchdog seeds peak from current RSS at entry
    # git_sha is either "unknown" (CI without .git) or a 40-char hex SHA.
    assert attrs["git_sha"] == "unknown" or len(attrs["git_sha"]) == 40


def test_train_errors_when_source_path_missing(runner: CliRunner, tmp_path: Path) -> None:
    config = tmp_path / "base.toml"
    config.write_text("[data]\ntarget_column = 'y'\n")  # source_path absent
    result = runner.invoke(app, ["--config", str(config), "train"])
    assert result.exit_code != 0


# ---------- PR-024: time_ordered dispatch path ----------


@pytest.fixture
def time_ordered_train_workdir(tmp_path: Path) -> Path:
    """Synthetic time-series Parquet + base.toml with ``data.split_kind = 'time_ordered'``.

    The ``ts`` column is intentionally out-of-order so the temporal sort step
    has work to do; ``y`` is generated from features so the trained model
    beats chance on the chronologically-last test fold.
    """
    rng = np.random.default_rng(0)
    n = 200
    ts = rng.permutation(n).tolist()  # shuffled timestamps
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    df = pl.DataFrame({"ts": ts, "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()})
    src = tmp_path / "synth_ts.parquet"
    df.write_parquet(src)

    storage_path = tmp_path / "studies" / "studies.db"
    config = tmp_path / "base.toml"
    config.write_text(
        f"""
[data]
source_path = "{src}"
target_column = "y"
cas_root = "{tmp_path}/cas"
manifests_root = "{tmp_path}/manifests"
split_kind = "time_ordered"
time_column = "ts"

[features]
spec = {{ numeric_columns = ["ts", "x1", "x2"], categorical_columns = [] }}

[training]
kind = "xgboost"
device = "cpu"
metric = "auc"
n_estimators = 16
max_depth = 3
learning_rate = 0.3

[runs]
storage_url = "sqlite:///{storage_path}"
artifacts_root = "{tmp_path}/studies/artifacts"
"""
        + repo_oracle_toml()
    )
    return tmp_path


def test_train_time_ordered_path_end_to_end_cpu(
    runner: CliRunner, time_ordered_train_workdir: Path
) -> None:
    """Dispatcher routes through ``temporal_train_val_test_split`` when
    ``cfg.data.split_kind == 'time_ordered'``. Confirms the CLI surface +
    config validator + dispatcher all wire correctly."""
    result = runner.invoke(
        app,
        _argv(time_ordered_train_workdir, "train"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "score (auc):" in result.stdout
    score_line = next(line for line in result.stdout.splitlines() if "score (auc):" in line)
    score = float(score_line.split(":")[-1].strip())
    assert 0.0 <= score <= 1.0


def test_train_time_ordered_rejects_missing_time_column(runner: CliRunner, tmp_path: Path) -> None:
    """PR-024 cross-field validator: ``time_ordered`` without ``time_column`` fails."""
    config = tmp_path / "base.toml"
    config.write_text(
        '[data]\nsource_path = "/nonexistent"\ntarget_column = "y"\n'
        'split_kind = "time_ordered"\n'  # no time_column
    )
    result = runner.invoke(app, ["--config", str(config), "train"])
    assert result.exit_code != 0
    # Validator raises pydantic ValidationError; typer wraps it as the runtime
    # exception. Inspect ``result.exception`` for the validator's text.
    assert result.exception is not None
    assert "time_ordered" in str(result.exception)
    assert "time_column" in str(result.exception)


def test_train_temporal_cv_rejects_random_split(runner: CliRunner, tmp_path: Path) -> None:
    """PR-024 cross-field validator: temporal CV + random split fails fast."""
    config = tmp_path / "base.toml"
    config.write_text(
        '[data]\nsource_path = "/nonexistent"\ntarget_column = "y"\n\n[cv]\nkind = "time_series"\n'
    )
    result = runner.invoke(app, ["--config", str(config), "train"])
    assert result.exit_code != 0
    assert result.exception is not None
    assert "time_series" in str(result.exception)
    assert "split_kind" in str(result.exception)


def test_train_uploads_diagnostics_to_artifact_store(
    runner: CliRunner, train_workdir: Path
) -> None:
    """PR-034: ``rux-ml train`` uploads ``metrics.json`` + ``fold_meta.json``.

    Asserts the per-study sub-directory exists under ``runs.artifacts_root``
    after a successful baseline run, and that the trial's
    ``system_attrs["artifacts:..."]`` entries (Optuna's auto-persistence)
    match the expected 2-file payload.
    """
    import optuna  # noqa: PLC0415

    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout

    storage_url = f"sqlite:///{train_workdir}/studies/studies.db"
    artifacts_root = train_workdir / "studies" / "artifacts"
    # Exactly one study should exist; it gets its own subdir per Optuna 4.8 FAQ.
    study_dirs = [p for p in artifacts_root.iterdir() if p.is_dir()]
    assert len(study_dirs) == 1, f"expected 1 study dir, got {study_dirs}"
    # Two artifact files per trial (metrics.json + fold_meta.json), flat uuid4 layout.
    artifact_files = list(study_dirs[0].iterdir())
    assert len(artifact_files) == 2, f"expected 2 artifacts, got {artifact_files}"
    # Auto-persisted in system_attrs (Probe 3 binding-at-creation).
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_url)
    artifact_keys = [k for k in study.trials[0].system_attrs if k.startswith("artifacts:")]
    assert len(artifact_keys) == 2


def test_train_use_native_smoke(runner: CliRunner, train_workdir: Path) -> None:
    """``--set training.use_native=true`` routes through ``XGBoostNativeAdapter`` (PR-033).

    Exit-code-0 + provenance recording verify the native path is wired
    end-to-end through ``cli/train._fit_and_score``; the "native API" log
    line in stderr confirms the branch was taken (vs. silently falling
    back to the sklearn-wrapper path).
    """
    result = runner.invoke(
        app,
        _argv(train_workdir, "--set", "training.use_native=true", "train"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "score (auc):" in result.stdout
    assert "native API" in result.stderr, (
        f"expected 'native API' ingest path log in stderr; got: {result.stderr!r}"
    )


@pytest.mark.gpu
def test_train_runs_end_to_end_gpu(runner: CliRunner, train_workdir: Path) -> None:
    """Same flow as the CPU test, just with ``--set training.device=cuda``."""
    result = runner.invoke(
        app,
        _argv(train_workdir, "--set", "training.device=cuda", "train"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "score (auc):" in result.stdout


# ---------- PR-041: the other labels ride as diagnostics ----------


def test_train_records_label_diagnostics_in_fold_meta(
    runner: CliRunner, train_workdir: Path
) -> None:
    """``[m9] diagnostic_columns`` are summarised per one-off fold, never trained on."""
    src = train_workdir / "synth.parquet"
    df = pl.read_parquet(src)
    df = df.with_columns((pl.col("x1") * 2.0).alias("y__other"))
    df.write_parquet(src)
    config = train_workdir / "base.toml"
    config.write_text(config.read_text() + '\n[m9]\ndiagnostic_columns = ["y__other"]\n')

    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout

    storage = f"sqlite:///{train_workdir}/studies/studies.db"
    summary = optuna.get_all_study_summaries(storage=storage)[0]
    metas = list_trial_artifacts(storage, summary.study_name, 0)
    fold_meta_id = next(m.artifact_id for m in metas if m.filename == "fold_meta.json")
    cfg = RuxMLConfig.from_layers(config)
    store = make_artifact_store(cfg, study_name=summary.study_name)
    out = train_workdir / "fold_meta.json"
    download_artifact(artifact_store=store, artifact_id=fold_meta_id, file_path=str(out))
    entry = json.loads(out.read_text())[0]
    diag = entry["diagnostics"]
    assert set(diag) == {"train", "val"}
    assert set(diag["train"]) == {"y__other"}
    stats = diag["train"]["y__other"]
    assert set(stats) == {"n", "null_count", "mean", "std", "min", "max"}
    assert stats["n"] + diag["val"]["y__other"]["n"] == int(200 * 0.85)
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_train_refuses_a_diagnostic_column_missing_from_the_set(
    runner: CliRunner, train_workdir: Path
) -> None:
    config = train_workdir / "base.toml"
    config.write_text(config.read_text() + '\n[m9]\ndiagnostic_columns = ["y__absent"]\n')
    result = runner.invoke(app, _argv(train_workdir, "train"))
    assert result.exit_code == 2, result.stdout
    assert "y__absent" in (result.stderr or result.stdout)


def test_label_diagnostics_count_nulls_apart_and_never_report_a_missing_mean_as_zero() -> None:
    """A diagnostic label may be null on rows where it is undefined (a markout on an
    unfilled rung, program D37 #2 "conditional on fill"). ``n`` counts values only,
    ``null_count`` counts the rest (NaN included), and an all-null column reports
    ``None`` statistics rather than a fabricated 0.0."""
    train = pl.DataFrame({"y__a": [1.0, None, 3.0, float("nan")], "y__b": [None, None, None, None]})
    val = pl.DataFrame({"y__a": [2.0, 4.0], "y__b": [0.5, None]})
    out = _label_diagnostics({"train": train, "val": val}, ["y__a", "y__b"])
    a = out["train"]["y__a"]
    # hand-computed over the two values {1, 3}: mean 2, sample std sqrt(2)
    assert a == {
        "n": 2.0,
        "null_count": 2.0,
        "mean": 2.0,
        "std": pytest.approx(2.0**0.5),
        "min": 1.0,
        "max": 3.0,
    }
    assert out["train"]["y__b"] == {
        "n": 0.0,
        "null_count": 4.0,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
    }
    b = out["val"]["y__b"]
    assert (b["n"], b["null_count"], b["mean"], b["min"], b["max"]) == (1.0, 1.0, 0.5, 0.5, 0.5)
    assert b["std"] is None  # one value has no sample std


# ---------- program PR-024 A9: a null target never reaches XGBoost ----------


def test_row_filter_keeps_null_targets_out_of_the_xgboost_fit(
    runner: CliRunner, train_workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label undefined on a row (the markout of an unfilled order is null, program
    PR-024 A9) is dropped by ``[m9] row_filter_non_null`` before the split, so neither
    the fitted nor the early-stopping target XGBoost receives holds a missing value."""
    src = train_workdir / "synth.parquet"
    df = pl.read_parquet(src).with_row_index("i")
    null_rows = df["i"] % 5 == 0  # 40 of the 200 rows lose their target
    df = df.with_columns(pl.when(null_rows).then(None).otherwise(pl.col("y")).alias("y"))
    df.drop("i").write_parquet(src)
    config = train_workdir / "base.toml"
    config.write_text(config.read_text() + '\n[m9]\nrow_filter_non_null = ["y"]\n')

    seen: list[np.ndarray] = []
    original_fit = XGBClassifier.fit

    def spy(self: XGBClassifier, x: object, y: object, **kw: object) -> object:
        seen.append(np.asarray(y, dtype=float))
        for _, y_eval in kw.get("eval_set") or []:  # type: ignore[union-attr]
            seen.append(np.asarray(y_eval, dtype=float))
        return original_fit(self, x, y, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(XGBClassifier, "fit", spy)
    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout
    assert len(seen) == 2  # the fitted target and the eval-set target
    assert not any(np.isnan(y).any() for y in seen)
    # 160 kept rows: train int(160 * 0.7) = 112, val int(160 * 0.15) = 24
    assert [y.size for y in seen] == [112, 24]


# ---------- PR-042: the split definition is recorded with every one-off fit ----------


def _fold_meta(train_workdir: Path) -> dict:
    storage = f"sqlite:///{train_workdir}/studies/studies.db"
    summary = optuna.get_all_study_summaries(storage=storage)[0]
    metas = list_trial_artifacts(storage, summary.study_name, 0)
    fold_meta_id = next(m.artifact_id for m in metas if m.filename == "fold_meta.json")
    cfg = RuxMLConfig.from_layers(train_workdir / "base.toml")
    store = make_artifact_store(cfg, study_name=summary.study_name)
    out = train_workdir / "fold_meta.json"
    download_artifact(artifact_store=store, artifact_id=fold_meta_id, file_path=str(out))
    return json.loads(out.read_text())[0]


def test_train_records_the_symbol_holdout_groups_per_partition(
    runner: CliRunner, train_workdir: Path
) -> None:
    """C9 'the fold definitions committed': the coin lists of a symbol-holdout fit."""
    import hashlib  # noqa: PLC0415

    src = train_workdir / "synth.parquet"
    coins = [f"C{i:02d}" for i in range(40)]
    pl.read_parquet(src).with_columns(
        pl.Series("coin", [coins[i % 40] for i in range(200)])
    ).write_parquet(src)
    config = train_workdir / "base.toml"
    text = config.read_text().replace(
        'target_column = "y"\n',
        'target_column = "y"\nsplit_kind = "symbol_holdout"\ngroup_column = "coin"\n'
        "symbol_holdout_seed = 1\n",
    )
    config.write_text(text)
    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout

    split = _fold_meta(train_workdir)["split_definition"]
    expected: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for c in coins:  # the documented contract, computed independently
        u = int.from_bytes(hashlib.sha256(f"1:{c}".encode()).digest()[:8], "big") / 2**64
        expected["train" if u < 0.7 else "val" if u < 0.85 else "test"].append(c)
    assert split == {
        "kind": "symbol_holdout",
        "group_column": "coin",
        "symbol_holdout_seed": 1,
        "groups": expected,
        "rows": {k: 5 * len(v) for k, v in expected.items()},
    }


def test_train_records_the_embargo_and_the_rows_it_purged(
    runner: CliRunner, train_workdir: Path
) -> None:
    src = train_workdir / "synth.parquet"
    pl.read_parquet(src).with_columns(pl.Series("t", list(range(200)))).write_parquet(src)
    config = train_workdir / "base.toml"
    config.write_text(
        config.read_text().replace(
            'target_column = "y"\n',
            'target_column = "y"\nsplit_kind = "time_ordered"\ntime_column = "t"\n'
            "split_embargo = 5\n",
        )
    )
    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout
    split = _fold_meta(train_workdir)["split_definition"]
    # 200 rows: train 0..139, val 140..169, test 170..199; embargo 5 purges t in
    # 135..139 from train and 165..169 from val -> 10 rows.
    assert split == {
        "kind": "time_ordered",
        "time_column": "t",
        "split_embargo": 5,
        "rows": {"train": 135, "val": 25, "test": 30},
        "purged_rows": 10,
    }
