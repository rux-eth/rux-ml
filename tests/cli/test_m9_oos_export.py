"""PR-055 (program PR-027 A8 / Q7, build step R3): an [m9] fit exports its val / test rows.

Every ``rux-ml train`` of an M9 problem writes ``oos_rows.parquet`` beside the trial's
record (an Optuna artifact of the trial, like ``fold_meta.json``): per val / test row the
``[m9] row_key_columns``, the prediction the fit scored and the realized target. Its
sha256 is in the trial record (``fold_meta.json`` ``oos_export`` and the user attribute
``oos_export_sha256``). The harness fits PR-026's floor on the model's TRAIN partition —
the view's rows that pass the recorded row filter, minus the exported keys, minus the
purged rows the split record names — and scores it on the SAME test rows. These tests hold
the export to the in-memory partitions (``make_splits`` on the whole frame), to the fit's
own predictions and scores, and prove that documented train-mask rule row for row.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import polars as pl
import pytest
from optuna.artifacts import download_artifact
from polars.testing import assert_frame_equal

from rux_ml._internal.seeds import make_seed_bag_from_hex
from rux_ml.cli import app
from rux_ml.config import RuxMLConfig
from rux_ml.data import load_parquet, make_splits, materialize
from rux_ml.data.partitions import FOLDS, SourceBatches, plan_partitions
from rux_ml.runs import make_artifact_store
from rux_ml.runs.artifacts import list_trial_artifacts
from rux_ml.training import compute_score
from rux_ml.training.xgboost.batches import Scored, category_levels, fit_batches, predict_fold
from tests.cli.conftest import REPO, m9_argv
from tests.cli.test_m9_train_cli import _fold_meta  # pyright: ignore[reportPrivateUsage]
from tests.conftest import m9_gates_overrides

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import CliRunner

PROBLEM = "m9_fill_frac"
TARGET = "y__fill_frac"
STUDIES = ["m9_regime_time_block", "m9_regime_row_random", "m9_regime_symbol_holdout"]


def _train(runner: CliRunner, tmp: Path, c6: Path, gates: dict[str, str], *extra: str) -> None:
    argv = m9_argv(tmp, c6, PROBLEM, *m9_gates_overrides(gates), *extra, "train")
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output


def _trial(tmp: Path) -> tuple[str, dict[str, Any]]:
    storage = f"sqlite:///{tmp}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    trial = optuna.load_study(study_name=summary.study_name, storage=storage).trials[0]
    return summary.study_name, trial.user_attrs


def _metrics(tmp: Path, study: str) -> dict[str, float]:
    storage = f"sqlite:///{tmp}/studies/studies.db"
    meta = next(m for m in list_trial_artifacts(storage, study, 0) if m.filename == "metrics.json")
    return json.loads(_download(tmp, study, meta.artifact_id, "metrics.json").read_text())


def _download(tmp: Path, study: str, artifact_id: str, name: str) -> Path:
    cfg = _cfg(tmp, None, {"runs.artifacts_root": str(tmp / "studies" / "artifacts")})
    out = tmp / f"downloaded-{name}"
    store = make_artifact_store(cfg, study_name=study)
    download_artifact(artifact_store=store, artifact_id=artifact_id, file_path=str(out))
    return out


def _cfg(c6: Path, study: str | None, overrides: dict[str, Any] | None = None) -> RuxMLConfig:
    sets: dict[str, Any] = {
        "data.source_path": str(c6 / "fill"),
        "training.device": "cpu",
        **(overrides or {}),
    }
    return RuxMLConfig.from_layers(
        REPO / "configs" / "base.toml",
        problem=PROBLEM,
        study=study,
        problems_dir=REPO / "configs" / "problems",
        studies_dir=REPO / "configs" / "studies",
        overrides=sets,
    )


def _export(tmp: Path) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any], bytes]:
    """The trial's export (as read back from the artifact store), its fold_meta entry,
    the trial's user attributes and the file's bytes."""
    study, attrs = _trial(tmp)
    storage = f"sqlite:///{tmp}/studies/studies.db"
    meta = _fold_meta(tmp, PROBLEM)
    record = meta["oos_export"]
    (art,) = [
        m for m in list_trial_artifacts(storage, study, 0) if m.filename == record["filename"]
    ]
    assert art.artifact_id == record["artifact_id"]
    path = _download(tmp, study, art.artifact_id, record["filename"])
    return pl.read_parquet(path), meta, attrs, path.read_bytes()


def _sorted(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return frame.sort(by, nulls_last=True, maintain_order=True)


@pytest.mark.parametrize("study", STUDIES)
def test_an_m9_train_exports_its_val_and_test_rows(
    study: str, runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    _train(runner, tmp_path, c6_set, signed_m9_gates, "--study", study)
    export, meta, attrs, raw = _export(tmp_path)
    record = meta["oos_export"]
    cfg = _cfg(c6_set, study)
    assert cfg.m9 is not None
    keys = list(cfg.m9.row_key_columns)

    # the file, its sha256 in the trial record (fold_meta + the user attribute)
    sha = hashlib.sha256(raw).hexdigest()
    assert record["sha256"] == sha == attrs["oos_export_sha256"]
    assert record["bytes"] == len(raw)
    assert record["schema_version"] == 1
    assert record["key_columns"] == keys
    assert record["target_column"] == TARGET
    assert export.columns == record["columns"] == ["partition", *keys, "prediction", "realized"]
    assert export.schema["partition"] == pl.String
    assert export.schema["prediction"] == pl.Float32
    assert export.schema["realized"] == pl.Float64

    # the rows: exactly the in-memory val / test partitions' keys and targets
    split_seed = make_seed_bag_from_hex(attrs["entropy_hex"]).split_seed
    assert cfg.data.source_path is not None
    parts = make_splits(
        cfg,
        materialize(load_parquet(cfg.data.source_path, oracle=cfg.data.oracle)),
        seed=split_seed,
    )
    assert record["rows"] == {"val": parts["val"].height, "test": parts["test"].height}
    assert record["rows"] == {k: meta["split_definition"]["rows"][k] for k in ("val", "test")}
    for fold in ("val", "test"):
        got = export.filter(pl.col("partition") == fold).select([*keys, "realized"])
        want = parts[fold].select(
            *keys, pl.col(TARGET).cast(pl.Float64).fill_null(float("nan")).alias("realized")
        )
        assert_frame_equal(_sorted(got, [*keys, "realized"]), _sorted(want, [*keys, "realized"]))

    # the predictions: the fit's own, in the batches' order, and its recorded scores
    bag = make_seed_bag_from_hex(attrs["entropy_hex"])
    source = SourceBatches(
        cfg.data.source_path, oracle=cfg.data.oracle, row_filter=cfg.m9.row_filter_non_null
    )
    plan = plan_partitions(cfg, source, seed=bag.split_seed)
    cats = category_levels(source, plan, cfg.features)
    fit = fit_batches(cfg, source, plan, TARGET, seed=bag.xgb_seed, categories=cats)
    spec = cfg.features.spec
    for fold in ("val", "test"):
        pred, y = predict_fold(fit, source, plan, FOLDS.index(fold), spec, TARGET, cats)
        rows = export.filter(pl.col("partition") == fold)
        assert rows["prediction"].to_numpy().tobytes() == pred.tobytes()  # bit for bit
        np.testing.assert_array_equal(rows["realized"].to_numpy(), y)
    val = export.filter(pl.col("partition") == "val")
    metric = cfg.training.metric
    val_score = compute_score(
        metric, Scored(), val["prediction"].to_numpy(), val["realized"].to_numpy()
    )
    assert _metrics(tmp_path, study_name_of(tmp_path))[f"fold_0_{metric}"] == val_score
    test = export.filter((pl.col("partition") == "test") & pl.col("realized").is_not_nan())
    test_score = compute_score(
        metric, Scored(), test["prediction"].to_numpy(), test["realized"].to_numpy()
    )
    assert meta["oos"]["score"] == test_score


def study_name_of(tmp: Path) -> str:
    return _trial(tmp)[0]


def _train_by_the_documented_rule(
    view: pl.DataFrame, export: pl.DataFrame, split: dict[str, Any], keys: list[str], time: str
) -> pl.DataFrame:
    """The harness's train mask (ARCHITECTURE § the out-of-sample row export): the view's
    rows passing the recorded row filter, minus every exported key, minus the purged rows."""
    for c in split.get("row_filter_non_null", []):
        present = pl.col(c).is_not_null()
        view = view.filter(
            present & pl.col(c).is_not_nan() if view.schema[c].is_float() else present
        )
    view = view.join(export.select(keys).unique(), on=keys, how="anti", nulls_equal=True)
    if split["kind"] == "time_ordered":
        view = view.filter(pl.col(time) + split["split_embargo"] < split["cuts"]["val_first_stamp"])
    if split.get("train_prefix_last_stamp") is not None:
        view = view.filter(pl.col(time) <= split["train_prefix_last_stamp"])
    return view


@pytest.mark.parametrize(
    ("study", "prefix"),
    [
        ("m9_regime_time_block", None),
        ("m9_regime_row_random", None),
        ("m9_regime_symbol_holdout", None),
        ("m9_regime_time_block", 0.5),
        ("m9_regime_row_random", 0.5),
    ],
)
def test_the_train_partition_is_the_documented_complement_of_the_export(
    study: str,
    prefix: float | None,
    runner: CliRunner,
    c6_set: Path,
    tmp_path: Path,
    signed_m9_gates: dict[str, str],
) -> None:
    extra = ["--study", study]
    if prefix is not None:
        extra += ["--set", f"data.train_prefix_frac={prefix}"]
    _train(runner, tmp_path, c6_set, signed_m9_gates, *extra)
    export, meta, attrs, _raw = _export(tmp_path)
    split = meta["split_definition"]
    cfg = _cfg(c6_set, study, {} if prefix is None else {"data.train_prefix_frac": prefix})
    assert cfg.m9 is not None and cfg.data.source_path is not None
    assert cfg.data.time_column is not None
    keys = list(cfg.m9.row_key_columns)
    view = materialize(load_parquet(cfg.data.source_path, oracle=cfg.data.oracle))
    parts = make_splits(cfg, view, seed=make_seed_bag_from_hex(attrs["entropy_hex"]).split_seed)
    got = _train_by_the_documented_rule(view, export, split, keys, cfg.data.time_column)
    assert got.height == parts["train"].height == split["rows"]["train"]
    assert_frame_equal(_sorted(got, view.columns), _sorted(parts["train"], view.columns))


def test_an_m9_fit_without_row_keys_is_refused_before_it_fits(
    runner: CliRunner,
    c6_set: Path,
    tmp_path: Path,
    signed_m9_gates: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rux_ml.cli import train as train_cli  # noqa: PLC0415

    def no_fit(*_a: object, **_k: object) -> None:
        msg = "the fit ran although the export could not be written"
        raise AssertionError(msg)

    monkeypatch.setattr(train_cli, "fit_batches", no_fit)
    argv = m9_argv(
        tmp_path, c6_set, PROBLEM, *m9_gates_overrides(signed_m9_gates),
        "--study", "m9_regime_time_block", "--set", "m9.row_key_columns=[]", "train",
    )  # fmt: skip
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "row_key_columns" in result.output
