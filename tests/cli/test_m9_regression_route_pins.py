"""PR-060: the M9 regression route is unchanged by the in-memory export and the val-less mode.

These pins pass on ``dev`` @ ``09b9bd5`` and after PR-060. The three M9 problems are
XGBoost regression fits (Brier / MAE), so they take the per-day batch path (PR-054) and
never reach PR-060's code; their configs' ``root_cfg_hash`` cannot move (PR-060 adds no
config field); a regression export carries schema v1's bytes — no ``prediction`` key in its
metadata (that key marks a probability export only) and no ``told_partition`` in
``fold_meta.json``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from rux_ml.cli import app
from rux_ml.config import RuxMLConfig
from rux_ml.config.root import cfg_hash
from rux_ml.runs.oos_export import OOS_EXPORT_METADATA_KEY, OosExportWriter
from rux_ml.training.xgboost.batches import uses_batch_fit
from tests.cli.conftest import REPO, m9_argv
from tests.cli.test_m9_oos_export import _export  # pyright: ignore[reportPrivateUsage]
from tests.conftest import m9_gates_overrides

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import CliRunner

# root_cfg_hash of each repo M9 problem (no study = its own time block), computed on
# dev @ 09b9bd5 (sha256 over canonical JSON — no path or platform enters it).
M9_ROOT_CFG_HASHES: dict[tuple[str, str | None], str] = {
    ("m9_fill_frac", None): ("fedfb4e47bcc5e2b4f2463f0d04c9cec6739570c9e0b2e9cc87a56a9a9ef7be4"),
    ("m9_fill_frac", "m9_regime_time_block"): (
        "fedfb4e47bcc5e2b4f2463f0d04c9cec6739570c9e0b2e9cc87a56a9a9ef7be4"
    ),
    ("m9_fill_frac", "m9_regime_row_random"): (
        "6defb35be2d308fc4213dd9c3afd7ad9faf3fe9ae173743392a11f82c49fae35"
    ),
    ("m9_fill_frac", "m9_regime_symbol_holdout"): (
        "56be2594fe7168618ea93805613f68bfb3d44fb3b7fbd81d466a2d6d5f78b01a"
    ),
    ("m9_markout_bp", None): ("22699caa6889f716f5cd48f8ba08953f61159b69573de74b7d093a541e875cdd"),
    ("m9_markout_bp", "m9_regime_time_block"): (
        "22699caa6889f716f5cd48f8ba08953f61159b69573de74b7d093a541e875cdd"
    ),
    ("m9_markout_bp", "m9_regime_row_random"): (
        "88bb73c386c0630c662e55e33c2cd69393fadd46fd3b5efdba9bb0d3909ca3c0"
    ),
    ("m9_markout_bp", "m9_regime_symbol_holdout"): (
        "fee42413223f8382506cf085201a1e9960f5ea68d814eedee7736a136cde386e"
    ),
    ("m9_walk_bp", None): ("50dab01cea0b96f280933bab52c580bbe8b620aa38433929dd0efc01506e05d5"),
    ("m9_walk_bp", "m9_regime_time_block"): (
        "50dab01cea0b96f280933bab52c580bbe8b620aa38433929dd0efc01506e05d5"
    ),
    ("m9_walk_bp", "m9_regime_row_random"): (
        "4ce8853065c9b0d359b065dbb0d5d2c734c358901f91ac3ea656074ee1ab5cc1"
    ),
    ("m9_walk_bp", "m9_regime_symbol_holdout"): (
        "f63d52d8ba07c1bac6dea671e2568111c7ba146ae0180ab7a9fda420036a6ff9"
    ),
}
SCHEMA_V1_METADATA_KEYS = {
    "schema_version", "columns", "key_columns", "target_column", "partitions",
    "split_definition",
}  # fmt: skip


@pytest.mark.parametrize(("problem", "study"), sorted(M9_ROOT_CFG_HASHES, key=str))
def test_the_m9_problems_stay_on_the_batch_path_with_their_config_hashes(
    problem: str, study: str | None
) -> None:
    cfg = RuxMLConfig.from_layers(
        REPO / "configs" / "base.toml",
        problem=problem,
        study=study,
        problems_dir=REPO / "configs" / "problems",
        studies_dir=REPO / "configs" / "studies",
    )
    assert uses_batch_fit(cfg)  # below the batch dispatch: PR-060's code never runs
    assert cfg.data.split_ratios["val"] > 0
    assert cfg_hash(cfg) == M9_ROOT_CFG_HASHES[(problem, study)]


def _v1_bytes(path: Path, chunks: list[tuple[str, pl.DataFrame, Any, Any]], meta: str) -> bytes:
    """Schema v1 written by hand (PR-055's recipe), independently of the writer."""
    writer: pq.ParquetWriter | None = None
    for partition, keys, pred, real in chunks:
        frame = keys.select(pl.lit(partition, dtype=pl.String).alias("partition"), *keys.columns)
        frame = frame.with_columns(
            pl.Series("prediction", pred, dtype=pl.Float32),
            pl.Series("realized", real, dtype=pl.Float64),
        )
        table = frame.to_arrow(compat_level=pl.CompatLevel.oldest())
        if writer is None:
            schema = table.schema.with_metadata({OOS_EXPORT_METADATA_KEY: meta})
            writer = pq.ParquetWriter(path, schema)
        writer.write_table(table.cast(schema))  # pyright: ignore[reportUnknownMemberType, reportPossiblyUnbound]
    assert writer is not None
    writer.close()
    return path.read_bytes()


def test_a_regression_export_is_written_byte_for_byte_as_schema_v1(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    split = {"kind": "time_ordered", "rows": {"train": 7, "val": 3, "test": 4}}
    chunks = [
        (
            part,
            pl.DataFrame({"stamp_ms": list(range(n)), "coin": ["BTC"] * n}),
            rng.normal(size=n).astype(np.float32),
            np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
        )
        for part, n in (("val", 3), ("test", 4))
    ]
    writer = OosExportWriter(
        tmp_path / "w.parquet", key_columns=["stamp_ms", "coin"], target_column="y",
        split_definition=split,
    )  # fmt: skip
    for part, keys, pred, real in chunks:
        writer.write(part, keys, pred, real)
    record = writer.close()
    meta = json.dumps(
        {
            "schema_version": 1,
            "columns": ["partition", "stamp_ms", "coin", "prediction", "realized"],
            "key_columns": ["stamp_ms", "coin"],
            "target_column": "y",
            "partitions": ["val", "test"],
            "split_definition": split,
        },
        sort_keys=True,
        default=str,
    )
    assert (tmp_path / "w.parquet").read_bytes() == _v1_bytes(tmp_path / "v1.parquet", chunks, meta)
    assert set(record) == {
        "filename", "schema_version", "sha256", "bytes", "rows", "columns", "key_columns",
        "target_column",
    }  # fmt: skip


def test_an_m9_batch_fit_export_keeps_the_regression_meaning(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    argv = m9_argv(
        tmp_path, c6_set, "m9_fill_frac", *m9_gates_overrides(signed_m9_gates),
        "--study", "m9_regime_time_block", "train",
    )  # fmt: skip
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    _, meta, _, raw = _export(tmp_path)
    assert "told_partition" not in meta
    record = meta["oos_export"]
    assert "prediction" not in record
    assert record["schema_version"] == 1
    (tmp_path / "export.parquet").write_bytes(raw)
    metadata = pq.read_schema(tmp_path / "export.parquet").metadata
    assert set(json.loads(metadata[OOS_EXPORT_METADATA_KEY.encode()])) == SCHEMA_V1_METADATA_KEYS
