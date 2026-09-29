"""The out-of-sample row export of an ``[m9]`` fit (PR-055; program PR-027 A8 / Q7).

Every ``rux-ml train`` of an M9 problem (the per-day batch fit, PR-054) writes
``oos_rows.parquet`` beside the trial's record — uploaded as an artifact of the trial into
the study's artifact store, where ``metrics.json`` and ``fold_meta.json`` live — and records
its sha256 in the trial record (``fold_meta.json`` ``oos_export`` and the trial user
attribute ``oos_export_sha256``). It is the interface program PR-027's floor comparison
consumes: the harness fits PR-026's floor on the model's **train** partition only and
scores it on the **same** test rows the model was scored on.

**Schema (version 1)** — one row per val / test row of the fit, val first, each partition in
the batches' order (day files by name, then file order; the order carries no meaning):

- ``partition`` (String): ``"val"`` or ``"test"``.
- the ``[m9] row_key_columns``, in their configured order, with the source's dtypes (a
  Categorical / Enum key is written as String).
- ``prediction`` (Float32): what the fit scored — ``XGBModel.predict``'s value up to the
  best iteration (PR-054's ``predict_fold``).
- ``realized`` (Float64): the target column cast to float64, null → NaN (NaN rows are
  outside the test score, as in ``fold_meta.json`` ``oos``).

The file's key-value metadata ``rux_ml.oos_export`` repeats the schema version, the key and
target columns and the fit's ``split_definition``, so the file is self-describing.

**The train mask (the harness's rule).** A row of the same view is in the fit's train
partition iff (1) it passes ``split_definition.row_filter_non_null`` (every listed column
non-null, and not NaN when float); (2) its key is not in the export; (3) it is not purged:
for ``kind == "time_ordered"``, ``stamp + split_embargo < cuts.val_first_stamp`` (the
embargo before val — rows purged before test are neither val, test nor train, and fail
the same inequality); and for a prefix fit, ``stamp <= train_prefix_last_stamp``.
Symbol-holdout and row-random purge nothing else. Rows sharing a key share a partition
under every regime (one stamp, one coin, one keyed hash), so the key complement is exact.
``tests/cli/test_m9_oos_export.py`` proves this rule row for row against the partitions.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

OOS_EXPORT_FILENAME = "oos_rows.parquet"
OOS_EXPORT_SCHEMA_VERSION = 1
OOS_EXPORT_METADATA_KEY = "rux_ml.oos_export"
PARTITION_COLUMN = "partition"
PREDICTION_COLUMN = "prediction"
REALIZED_COLUMN = "realized"
PARTITIONS: tuple[str, str] = ("val", "test")
_HASH_CHUNK_BYTES = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OosExportWriter:
    """Streams the export one batch at a time (a ``pyarrow`` ``ParquetWriter``), so the
    rows are never held whole: at 156 days the fill fit's val + test are ≈ 28M rows."""

    def __init__(
        self,
        path: Path,
        *,
        key_columns: Sequence[str],
        target_column: str,
        split_definition: Mapping[str, Any],
    ) -> None:
        keys = list(key_columns)
        if not keys:
            msg = (
                "the [m9] out-of-sample export needs m9.row_key_columns: the keys are the "
                "interface the floor comparison masks the train partition with"
            )
            raise ValueError(msg)
        clash = {PARTITION_COLUMN, PREDICTION_COLUMN, REALIZED_COLUMN} & set(keys)
        if clash:
            msg = f"m9.row_key_columns {sorted(clash)} collide with the export's own columns"
            raise ValueError(msg)
        self.path = path
        self.key_columns = keys
        self.target_column = target_column
        self.columns = [PARTITION_COLUMN, *keys, PREDICTION_COLUMN, REALIZED_COLUMN]
        self._metadata = json.dumps(
            {
                "schema_version": OOS_EXPORT_SCHEMA_VERSION,
                "columns": self.columns,
                "key_columns": keys,
                "target_column": target_column,
                "partitions": list(PARTITIONS),
                "split_definition": dict(split_definition),
            },
            sort_keys=True,
            default=str,
        )
        self.rows: dict[str, int] = dict.fromkeys(PARTITIONS, 0)
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None

    def sink(
        self, partition: str
    ) -> Callable[[pl.DataFrame, NDArray[np.float32], NDArray[np.float64]], None]:
        """A ``predict_fold`` sink writing ``partition``'s rows."""
        if partition not in PARTITIONS:
            msg = f"the export holds {PARTITIONS}, not {partition!r}"
            raise ValueError(msg)

        def write(
            keys: pl.DataFrame, prediction: NDArray[np.float32], realized: NDArray[np.float64]
        ) -> None:
            self.write(partition, keys, prediction, realized)

        return write

    def write(
        self,
        partition: str,
        keys: pl.DataFrame,
        prediction: NDArray[np.float32],
        realized: NDArray[np.float64],
    ) -> None:
        frame = keys.select(
            pl.lit(partition, dtype=pl.String).alias(PARTITION_COLUMN),
            *[
                pl.col(c).cast(pl.String)
                if isinstance(keys.schema[c], (pl.Categorical, pl.Enum))
                else pl.col(c)
                for c in self.key_columns
            ],
        ).with_columns(
            pl.Series(PREDICTION_COLUMN, prediction, dtype=pl.Float32),
            pl.Series(REALIZED_COLUMN, realized, dtype=pl.Float64),
        )
        table = frame.to_arrow(compat_level=pl.CompatLevel.oldest())
        if self._writer is None:
            self._schema = table.schema.with_metadata({OOS_EXPORT_METADATA_KEY: self._metadata})
            self._writer = pq.ParquetWriter(self.path, self._schema)
        assert self._schema is not None
        self._writer.write_table(table.cast(self._schema))  # pyright: ignore[reportUnknownMemberType]
        self.rows[partition] += frame.height

    def close(self) -> dict[str, Any]:
        """Close the file; its record for ``fold_meta.json`` (the artifact id is added
        when the file is uploaded)."""
        if self._writer is None:
            msg = "the out-of-sample export has no rows: the fit had no val or test rows"
            raise ValueError(msg)
        self._writer.close()
        return {
            "filename": self.path.name,
            "schema_version": OOS_EXPORT_SCHEMA_VERSION,
            "sha256": file_sha256(self.path),
            "bytes": self.path.stat().st_size,
            "rows": dict(self.rows),
            "columns": self.columns,
            "key_columns": self.key_columns,
            "target_column": self.target_column,
        }
