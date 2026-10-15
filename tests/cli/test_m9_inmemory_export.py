"""PR-060 (program PR-082 leans 1 - 2): the in-memory ``[m9]`` path writes the per-row OOS
export and runs val-less.

An ``[m9]`` fit that is not a per-day batch fit (a classifier: no M9 problem is one, the
program's v0.5 probe cells are) keeps the in-memory path. It now writes the same
``oos_rows.parquet`` the batch path writes (PR-055): per val / test row the
``[m9] row_key_columns``, the prediction — ``predict_proba[:, 1]`` for a binary classifier
— and the realized target. With ``data.split_ratios.val = 0`` (the val-less mode) the fit
has no ``eval_set`` and no early stopping, and the told value is the TEST score, marked
``told_partition = "test"`` in ``fold_meta.json``. The tests use a made-up table only.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import polars as pl
import pyarrow.parquet as pq
import pytest
from optuna.artifacts import download_artifact
from polars.testing import assert_frame_equal
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from rux_ml._internal.seeds import SeedBag, make_seed_bag_from_hex
from rux_ml.cli import app
from rux_ml.cli.train import _fit_and_score  # pyright: ignore[reportPrivateUsage]
from rux_ml.config import RuxMLConfig
from rux_ml.data import load_parquet, make_splits, materialize
from rux_ml.features import make_features
from rux_ml.runs import list_trial_artifacts, make_artifact_store
from rux_ml.training import make_trainer
from tests.conftest import repo_oracle_toml

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import CliRunner

KEYS = ["stamp_ms", "coin"]
COINS = ["A", "B", "C", "D"]
N_STAMPS = 60
STEP_MS = 1_000
_VAL_LESS = "split_ratios = { train = 0.85, val = 0.0, test = 0.15 }\n"


def _write(
    tmp: Path,
    *,
    data: str = "",
    training: str = "",
    m9: str | None = 'row_key_columns = ["stamp_ms", "coin"]\nh_max_ms = 1000\n',
    classes: int = 2,
    test_target_nan: bool = False,
) -> Path:
    """A made-up time-ordered table (60 stamps x 4 coins) and its config."""
    rng = np.random.default_rng(0)
    n = N_STAMPS * len(COINS)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    signal = 0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)
    y = (signal > 0).astype(float) if classes == 2 else np.digitize(signal, [-0.5, 0.5]) * 1.0
    stamps = [i * STEP_MS for i in range(N_STAMPS) for _ in COINS]
    if test_target_nan:  # every row from stamp 45 s on: the whole test partition
        y = np.where(np.asarray(stamps) >= 45 * STEP_MS, np.nan, y)
    pl.DataFrame(
        {
            "stamp_ms": stamps,
            "coin": [c for _ in range(N_STAMPS) for c in COINS],
            "x1": x1,
            "x2": x2,
            "y": y,
        }
    ).write_parquet(tmp / "synth.parquet")
    text = f"""
[data]
source_path = "{tmp}/synth.parquet"
target_column = "y"
cas_root = "{tmp}/cas"
manifests_root = "{tmp}/manifests"
split_kind = "time_ordered"
time_column = "stamp_ms"
split_embargo = 1000
{data}
[features]
spec = {{ numeric_columns = ["x1", "x2"], categorical_columns = [] }}

[training]
kind = "xgboost"
device = "cpu"
metric = "logloss"
n_estimators = 16
max_depth = 3
learning_rate = 0.3
{training}
[runs]
storage_url = "sqlite:///{tmp}/studies/studies.db"
artifacts_root = "{tmp}/studies/artifacts"
"""
    if m9 is not None:
        text += "\n[m9]\n" + m9
    config = tmp / "base.toml"
    config.write_text(text + repo_oracle_toml())
    return config


def _train(runner: CliRunner, config: Path) -> Any:
    return runner.invoke(app, ["--config", str(config), "train"], catch_exceptions=False)


def _storage(tmp: Path) -> str:
    return f"sqlite:///{tmp}/studies/studies.db"


def _trial(tmp: Path) -> tuple[str, optuna.trial.FrozenTrial]:
    (summary,) = optuna.get_all_study_summaries(storage=_storage(tmp))
    study = optuna.load_study(study_name=summary.study_name, storage=_storage(tmp))
    return summary.study_name, study.trials[0]


def _artifact(tmp: Path, study: str, filename: str) -> Path:
    (meta,) = [m for m in list_trial_artifacts(_storage(tmp), study, 0) if m.filename == filename]
    cfg = RuxMLConfig.from_layers(tmp / "base.toml")
    out = tmp / f"downloaded-{filename}"
    store = make_artifact_store(cfg, study_name=study)
    download_artifact(artifact_store=store, artifact_id=meta.artifact_id, file_path=str(out))
    return out


def _fold_meta(tmp: Path) -> dict[str, Any]:
    study, _ = _trial(tmp)
    return json.loads(_artifact(tmp, study, "fold_meta.json").read_text())[0]


def _refit(config: Path, bag: SeedBag, *, val_less: bool) -> tuple[Any, dict[str, pl.DataFrame]]:
    """The fit, done independently of ``rux-ml train`` from the trial's seeds."""
    cfg = RuxMLConfig.from_layers(config)
    assert cfg.data.source_path is not None
    df = materialize(load_parquet(cfg.data.source_path, oracle=cfg.data.oracle))
    splits = make_splits(cfg, df, seed=bag.split_seed)
    pipeline = make_features(cfg.features, cardinalities=None)
    x_train, y_train = splits["train"].drop("y"), splits["train"]["y"]
    pipeline.fit(x_train, y_train.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    trainer = make_trainer(cfg.training, seed=bag.xgb_seed)
    kw: dict[str, Any] = {}
    if not val_less:
        x_val = pipeline.transform(splits["val"].drop("y")).to_pandas()  # pyright: ignore[reportUnknownMemberType]
        kw["eval_set"] = [(x_val, splits["val"]["y"].to_numpy())]
    trainer.fit(pipeline.transform(x_train).to_pandas(), y_train.to_numpy(), verbose=False, **kw)  # pyright: ignore[reportUnknownMemberType]
    return (trainer, pipeline), splits


def _proba(model: tuple[Any, Any], frame: pl.DataFrame) -> np.ndarray:
    trainer, pipeline = model
    x = pipeline.transform(frame.drop("y")).to_pandas()
    return np.asarray(trainer.predict_proba(x))[:, 1]


def _spy_fit(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    original = XGBClassifier.fit

    def spy(self: XGBClassifier, x: object, y: object, **kw: Any) -> object:
        seen.append(kw)
        return original(self, x, y, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(XGBClassifier, "fit", spy)
    return seen


def _no_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_fit(*_a: object, **_k: object) -> None:
        msg = "the fit ran although the run should have been refused before it"
        raise AssertionError(msg)

    monkeypatch.setattr(XGBClassifier, "fit", no_fit)


# ---------- the export on the in-memory path ----------


def test_an_inmemory_m9_classifier_exports_p_class_1_for_val_then_test_rows(
    runner: CliRunner, tmp_path: Path
) -> None:
    config = _write(tmp_path)
    result = _train(runner, config)
    assert result.exit_code == 0, result.output
    study, trial = _trial(tmp_path)
    meta = _fold_meta(tmp_path)
    record = meta["oos_export"]
    path = _artifact(tmp_path, study, record["filename"])
    raw = path.read_bytes()
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert trial.user_attrs["oos_export_sha256"] == record["sha256"]
    assert "told_partition" not in meta  # val > 0: the told value is the val score

    export = pl.read_parquet(path)
    assert export.columns == ["partition", *KEYS, "prediction", "realized"]
    assert export.schema["prediction"] == pl.Float32
    assert export.schema["realized"] == pl.Float64
    rows = meta["split_definition"]["rows"]
    assert rows["val"] > 0 and rows["test"] > 0
    assert export["partition"].to_list() == ["val"] * rows["val"] + ["test"] * rows["test"]
    assert record["rows"] == {"val": rows["val"], "test": rows["test"]}
    metadata = json.loads(pq.read_schema(path).metadata[b"rux_ml.oos_export"])
    assert metadata["prediction"] == "predict_proba[:, 1]"

    model, splits = _refit(
        config, make_seed_bag_from_hex(trial.user_attrs["entropy_hex"]), val_less=False
    )
    for part in ("val", "test"):
        got = export.filter(pl.col("partition") == part)
        assert_frame_equal(got.select(KEYS), splits[part].select(KEYS))
        want = _proba(model, splits[part])
        np.testing.assert_array_equal(got["prediction"].to_numpy(), want.astype(np.float32))
        np.testing.assert_array_equal(got["realized"].to_numpy(), splits[part]["y"].to_numpy())
    # the exported p-hat is what the test logloss scored
    p_test = _proba(model, splits["test"])
    assert meta["oos"]["score"] == pytest.approx(
        log_loss(splits["test"]["y"].to_numpy(), np.column_stack([1 - p_test, p_test])), rel=1e-12
    )
    assert trial.value == pytest.approx(
        log_loss(
            splits["val"]["y"].to_numpy(),
            np.column_stack([1 - (p := _proba(model, splits["val"])), p]),
        ),
        rel=1e-12,
    )


def test_the_export_leaves_the_val_gt_0_fit_unchanged(
    tmp_path: Path, seed_bag: SeedBag, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Program ACCEPTANCE C2's pin: the fit, its val score and its test record are the same
    with and without the export; the fit still sees exactly one eval_set."""
    config = _write(tmp_path)
    cfg = RuxMLConfig.from_layers(config)
    seen = _spy_fit(monkeypatch)
    src = tmp_path / "synth.parquet"
    plain = _fit_and_score(cfg, src, "y", seed_bag, None, export_dir=None)
    (tmp_path / "export").mkdir()
    exported = _fit_and_score(cfg, src, "y", seed_bag, None, export_dir=tmp_path / "export")
    assert [sorted(kw) for kw in seen] == [["eval_set", "verbose"]] * 2
    assert exported[0] == plain[0]  # the val score
    assert exported[1] == plain[1]  # best_iteration
    assert exported[4]["oos"] == plain[4]["oos"]
    assert "oos_export" not in plain[4]
    assert exported[4]["oos_export"]["rows"]["test"] > 0


# ---------- the val-less mode ----------


@pytest.mark.parametrize("native", [False, True], ids=["sklearn", "native"])
def test_a_val_less_fit_has_no_eval_set_and_tells_the_test_score(
    native: bool, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = "early_stopping_rounds = 0\n" + ("use_native = true\n" if native else "")
    config = _write(tmp_path, data=_VAL_LESS, training=training)
    seen = _spy_fit(monkeypatch)
    trains: list[dict[str, Any]] = []
    import xgboost  # noqa: PLC0415

    original_train = xgboost.train

    def spy_train(**kw: Any) -> Any:
        trains.append(kw)
        return original_train(**kw)

    monkeypatch.setattr("rux_ml.training.xgboost.native_adapter.xgb.train", spy_train)
    result = _train(runner, config)
    assert result.exit_code == 0, result.output
    if native:
        assert "native API" in result.output
        assert len(trains) == 1
        assert "evals" not in trains[0] and "early_stopping_rounds" not in trains[0]
    else:
        assert len(seen) == 1 and "eval_set" not in seen[0]

    study, trial = _trial(tmp_path)
    meta = _fold_meta(tmp_path)
    assert meta["told_partition"] == "test"
    told = meta["oos"]["score"]
    assert trial.value == told
    metrics = json.loads(_artifact(tmp_path, study, "metrics.json").read_text())
    assert metrics["fold_0_logloss"] == told
    line = next(s for s in result.stdout.splitlines() if s.startswith("score (logloss):"))
    assert float(line.split(":")[-1]) == pytest.approx(told, abs=5e-7)

    split = meta["split_definition"]
    assert split["rows"]["val"] == 0
    assert split["cuts"]["val_first_stamp"] == split["cuts"]["test_first_stamp"]
    record = meta["oos_export"]
    assert record["rows"] == {"val": 0, "test": split["rows"]["test"]}
    export = pl.read_parquet(_artifact(tmp_path, study, record["filename"]))
    assert set(export["partition"].to_list()) == {"test"}

    # the harness's train mask (runs/oos_export.py) marks the train partition exactly
    view = pl.read_parquet(tmp_path / "synth.parquet")
    exported = export.select(KEYS)
    cut = split["cuts"]["val_first_stamp"]
    masked = view.join(exported, on=KEYS, how="anti").filter(
        pl.col("stamp_ms") + split["split_embargo"] < cut
    )
    _, splits = _refit(
        config, make_seed_bag_from_hex(trial.user_attrs["entropy_hex"]), val_less=True
    )
    assert masked.height == split["rows"]["train"] == splits["train"].height
    assert_frame_equal(masked.sort(KEYS), splits["train"].sort(KEYS))
    if not native:  # the told value is the test logloss of the independent re-fit
        model, _ = _refit(
            config, make_seed_bag_from_hex(trial.user_attrs["entropy_hex"]), val_less=True
        )
        p = _proba(model, splits["test"])
        assert told == pytest.approx(
            log_loss(splits["test"]["y"].to_numpy(), np.column_stack([1 - p, p])), rel=1e-12
        )
        np.testing.assert_array_equal(export["prediction"].to_numpy(), p.astype(np.float32))


# ---------- the refusals, each by name and before the fit ----------


@pytest.mark.parametrize(
    ("case", "kwargs", "named"),
    [
        ("val_less_early_stopping_on", {"data": _VAL_LESS}, "early_stopping_rounds"),
        (
            "val_less_without_m9",
            {"data": _VAL_LESS, "training": "early_stopping_rounds = 0\n", "m9": None},
            "[m9]",
        ),
        ("keyless_m9", {"m9": "row_key_columns = []\nh_max_ms = 1000\n"}, "row_key_columns"),
        ("three_classes", {"classes": 3}, "classes"),
        (
            "val_less_no_realized_test_label",
            {
                "data": _VAL_LESS,
                "training": "early_stopping_rounds = 0\n",
                "test_target_nan": True,
            },
            "realized",
        ),
    ],
)
def test_the_route_refuses_by_name_before_the_fit(
    case: str,
    kwargs: dict[str, Any],
    named: str,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write(tmp_path, **kwargs)
    _no_fit(monkeypatch)
    result = runner.invoke(app, ["--config", str(config), "train"])
    assert result.exit_code == 2, (case, result.output)
    assert named in result.output, (case, result.output)
    if case.startswith("val_less_without") or case == "val_less_early_stopping_on":
        # refused at config, before the trial row exists
        assert optuna.get_all_study_summaries(storage=_storage(tmp_path)) == []


def test_promote_refuses_a_val_less_trial_by_name(
    tmp_path: Path, seed_bag: SeedBag, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision 6: promote's in-memory re-fit always fits against val; a val-less trial is
    refused by name before any data is read, never failed inside xgboost."""
    from rux_ml.registry import promote as promote_mod  # noqa: PLC0415

    config = _write(tmp_path, data=_VAL_LESS, training="early_stopping_rounds = 0\n")
    cfg = RuxMLConfig.from_layers(config)

    def no_read(*_a: object, **_k: object) -> None:
        msg = "promote read the source although the trial is refused"
        raise AssertionError(msg)

    monkeypatch.setattr(promote_mod, "load_parquet", no_read)
    with pytest.raises(promote_mod.PromoteValLessError) as info:
        promote_mod._refit(cfg, bag=seed_bag, hashed_files=[])  # pyright: ignore[reportPrivateUsage]
    assert isinstance(info.value, ValueError)  # exit 2, like every promote refusal
    assert "split_ratios.val = 0" in str(info.value)
