"""PR-053 on the harness-shaped set: the per-file logical hash equals the whole-source digest on
``c6_set``'s ``fill/`` and ``walk/`` subtrees, through the paths ``train`` (``data_hashes``) and the
bridge (``training_set_sidecar``, the value promote and ``data bridge`` compare) take."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from rux_ml.data import versioning
from rux_ml.data.bridge import training_set_sidecar
from rux_ml.data.loaders import iter_parquet_files
from rux_ml.runs import data_hashes
from tests.conftest import repo_oracle_cfg

if TYPE_CHECKING:
    from pathlib import Path

PINNED_POLARS = "1.40.1"
# Computed by the pre-PR-053 code on dev @ 9e3f0c3 under Polars 1.40.1.
PINNED_C6 = {
    "fill": "185d2b9cf3685f3544885bc30987fc67cddc137763bf06d61ebe472aced2d4d8",
    "walk": "81f710717d194c77357005b7a347a8f3bad1dc54627c904c8080adc9629bd4e7",
}


@pytest.mark.parametrize("subtree", ["fill", "walk"])
def test_the_c6_subtree_digest_is_unchanged_on_every_path(c6_set: Path, subtree: str) -> None:
    source = c6_set / subtree
    reference = versioning._logical_hash_whole_source(iter_parquet_files(source))
    assert versioning._logical_hash(iter_parquet_files(source)) == reference
    assert data_hashes(source, oracle=repo_oracle_cfg())["data_logical_hash"] == reference[0]
    assert (
        training_set_sidecar(source, oracle=repo_oracle_cfg())["data_logical_hash"] == reference[0]
    )
    if pl.__version__ == PINNED_POLARS:
        assert reference[0] == PINNED_C6[subtree]
