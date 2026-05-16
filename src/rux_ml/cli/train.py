"""``rux-ml train`` — single baseline training end-to-end (per PR-006 + D7).

Wraps a one-off training as a 1-trial Optuna study so the same SQLite store
holds both sweeps and baselines (per D7). The provenance triple subset
recorded here is the **PR-006 minimum** (config hashes + ``data_hash`` +
``git_sha``); ``entropy_hex`` + the full environment block land in PR-013.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import optuna
import polars as pl
import typer

from rux_ml._internal.git import git_sha
from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig, cfg_hash, layer_cfg_hash
from rux_ml.data import compute_data_hash, load_parquet, materialize, train_val_test_split
from rux_ml.features import cardinalities_from, make_features
from rux_ml.training import (
    compute_score,
    estimate_x_bytes,
    make_trainer,
    optuna_direction,
    select_ingest,
)

if TYPE_CHECKING:
    from rux_ml.cli._shared import GlobalOptions

_DEFAULT_SPLIT_SEED = 0  # PR-013 will replace with SeedSequence-derived per-component seeds.
_HASH_LAYERS = ("data", "features", "training", "tuning", "runs", "registry", "memory")
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def _study_name(cfg: RuxMLConfig, opts: GlobalOptions) -> str:
    stamp = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
    return cfg.runs.study_name_template.format(
        problem=opts.problem or "default",
        study=opts.study or "oneoff",
        stamp=stamp,
    )


def _ensure_storage_parent(storage_url: str) -> None:
    """Create the parent dir for a ``sqlite:///…`` URL so Optuna can open it."""
    if not storage_url.startswith("sqlite:///"):
        return
    db_path = Path(storage_url.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _require(cfg: RuxMLConfig) -> tuple[Path, str]:
    if cfg.data.source_path is None or cfg.data.target_column is None:
        msg = (
            "rux-ml train requires data.source_path and data.target_column to be set "
            "(via configs/problems/<problem>.toml or --set data.source_path=… "
            "--set data.target_column=…)"
        )
        raise typer.BadParameter(msg)
    return cfg.data.source_path, cfg.data.target_column


def _strip_target(df: pl.DataFrame, target_col: str) -> tuple[pl.DataFrame, pl.Series]:
    return df.drop(target_col), df[target_col]


def _build_user_attrs(cfg: RuxMLConfig, data_hashes: dict[str, str]) -> dict[str, str]:
    attrs: dict[str, str] = {
        f"{layer}_cfg_hash": layer_cfg_hash(cfg, layer) for layer in _HASH_LAYERS
    }
    attrs["root_cfg_hash"] = cfg_hash(cfg)
    attrs["git_sha"] = git_sha()
    attrs.update(data_hashes)
    return attrs


def _data_hashes(source_path: Path) -> dict[str, str]:
    composite = compute_data_hash(source_path)
    return {
        "data_bytes_hash": composite["bytes_hash"],
        "data_logical_hash": composite["logical_hash"],
        # Composite ``data_hash`` matches snapshot()'s version_id derivation so
        # promoted runs can cross-reference snapshots by the same identifier.
        "data_hash": f"{composite['bytes_hash']}|{composite['logical_hash']}",
    }


def _fit_and_score(
    cfg: RuxMLConfig, source_path: Path, target_col: str
) -> tuple[float, int | None]:
    df = materialize(load_parquet(source_path))
    splits = train_val_test_split(df, ratios=cfg.data.split_ratios, seed=_DEFAULT_SPLIT_SEED)
    x_train, y_train = _strip_target(splits["train"], target_col)
    x_val, y_val = _strip_target(splits["val"], target_col)

    cards = cardinalities_from(x_train, cfg.features.spec.categorical_columns)
    pipeline = make_features(cfg.features, cardinalities=cards if cards else None)
    pipeline.fit(x_train, y_train.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    # sklearn Pipeline.transform stubs return ``Unknown``; PR-005's terminal
    # ``to_polars`` step guarantees a Polars DataFrame at the boundary.
    x_train_t = cast("pl.DataFrame", pipeline.transform(x_train))  # pyright: ignore[reportUnknownMemberType]
    x_val_t = cast("pl.DataFrame", pipeline.transform(x_val))  # pyright: ignore[reportUnknownMemberType]

    # Decision-rule classification: PR-006 logs the chosen DMatrix class so
    # operators see which path is in use. Actual construction in the in-memory
    # branch is delegated to the sklearn wrapper (XGBoost builds the
    # QuantileDMatrix internally for tree_method="hist"); the native ExtMem
    # branch executes via xgb.train + ParquetDataIter once a problem actually
    # exceeds the threshold (see training/ingest.py for the contract).
    dmatrix_cls = select_ingest(estimate_x_bytes(x_train_t), cfg.data)
    typer.echo(f"  ingest path: {dmatrix_cls.__name__}", err=True)

    trainer = make_trainer(cfg.training)
    x_train_pd = x_train_t.to_pandas()
    x_val_pd = x_val_t.to_pandas()
    trainer.fit(
        x_train_pd,
        y_train.to_numpy(),
        eval_set=[(x_val_pd, y_val.to_numpy())],
        verbose=False,
    )
    score = compute_score(cfg.training.metric, trainer, x_val_pd, y_val.to_numpy())
    best_iter = getattr(trainer, "best_iteration", None)
    return score, int(best_iter) if best_iter is not None else None


def run_command(ctx: typer.Context) -> None:
    """Run a single baseline training; recorded as a 1-trial Optuna study (per D7)."""
    opts = get_options(ctx)
    cfg = RuxMLConfig.from_layers(
        opts.config,
        problem=opts.problem,
        study=opts.study,
        overrides=opts.overrides,
    )
    source_path, target_col = _require(cfg)

    data_hashes = _data_hashes(source_path)
    score, best_iter = _fit_and_score(cfg, source_path, target_col)

    _ensure_storage_parent(cfg.runs.storage_url)
    study_name = _study_name(cfg, opts)
    study = optuna.create_study(
        study_name=study_name,
        storage=cfg.runs.storage_url,
        direction=optuna_direction(cfg.training.metric),
        load_if_exists=True,
    )
    trial = study.ask()
    for key, value in _build_user_attrs(cfg, data_hashes).items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("metric", cfg.training.metric)
    if best_iter is not None:
        trial.set_user_attr("best_iteration", best_iter)
    study.tell(trial, score)

    typer.echo(f"score ({cfg.training.metric}): {score:.6f}")
    typer.echo(f"  study:    {study_name}")
    typer.echo(f"  trial:    {trial.number}")
    typer.echo(f"  storage:  {cfg.runs.storage_url}")
    if best_iter is not None:
        typer.echo(f"  best_iter: {best_iter}")
