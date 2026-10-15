"""PR-053 on the harness-shaped set: the per-file logical hash equals the whole-source digest on
c6's ``fill/`` and ``walk/`` subtrees, through the paths ``train`` (``data_hashes``) and the
bridge (``training_set_sidecar``, the value promote and ``data bridge`` compare) take.

PR-061: the pins read committed bytes, ``tests/fixtures/c6_pin/{fill,walk}/``, not the generated
``c6_set``. ``_c6_day``'s gamma- and beta-fed columns differ at ULP level between machines: numpy's
arm64 wheel fuses ``random_standard_gamma``'s ``1 + c*X`` into one FMA and the x86_64 wheel does
not. The same bytes hash the same on both. The six files were drawn by ``_c6_day`` with the
``c6_set`` loop (``default_rng(0)``) on a Mac (numpy 2.4.5 arm64, Polars 1.40.1). They are a
record, not something to regenerate: only an FMA-contracting numpy build reproduces them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from rux_ml.data import versioning
from rux_ml.data.bridge import training_set_sidecar
from rux_ml.data.loaders import iter_parquet_files
from rux_ml.runs import data_hashes
from tests.conftest import repo_oracle_cfg

C6_PIN = Path(__file__).resolve().parents[1] / "fixtures" / "c6_pin"
PINNED_POLARS = "1.40.1"
# Each committed file's sha256, checked under any Polars version (program PR-061 Phase 3,
# pr061-p3/mac-hash.json): a change to the bytes fails here, before any digest is read.
PINNED_FILES = {
    "fill": {
        "2026-02-16.parquet": "4a9f5b21d46eff2fb2111cfa08a7dae81cfe2af954f1a0cd888d521b7c6f04d3",
        "2026-02-17.parquet": "0d0fcbc79af6315b89f65a9fed9246f031a2a41f5dc678e40f221055aa1b8005",
        "2026-02-18.parquet": "c49c9b30a357e15c029cf6ce9e638cd60c8b3ef9d9930514a518552a4eb7765f",
    },
    "walk": {
        "2026-02-16.parquet": "4cfddc6b59dce6d3955c361e1065969868f0ac652e5314caa7659d84a9bd94aa",
        "2026-02-17.parquet": "3c0d2e3a23b59ba81493d09d17fc155c3955b3d42f43b3721501cebede104da9",
        "2026-02-18.parquet": "48bc780cb309f6cf8ec6f8fb310297271aa23e0f75325fd2a715bfdc0024daf6",
    },
}
# Computed by the pre-PR-053 code on dev @ 9e3f0c3 under Polars 1.40.1.
PINNED_C6 = {
    "fill": "185d2b9cf3685f3544885bc30987fc67cddc137763bf06d61ebe472aced2d4d8",
    "walk": "81f710717d194c77357005b7a347a8f3bad1dc54627c904c8080adc9629bd4e7",
}


@pytest.mark.parametrize("subtree", ["fill", "walk"])
def test_the_committed_c6_bytes_are_the_pinned_files(subtree: str) -> None:
    source = C6_PIN / subtree
    assert source.is_dir(), f"the committed c6 fixture is missing: {source}"
    found = {
        f.relative_to(source).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in iter_parquet_files(source)
    }
    assert found == PINNED_FILES[subtree]


@pytest.mark.parametrize("subtree", ["fill", "walk"])
def test_the_c6_subtree_digest_is_unchanged_on_every_path(subtree: str) -> None:
    source = C6_PIN / subtree
    reference = versioning._logical_hash_whole_source(iter_parquet_files(source))
    assert versioning._logical_hash(iter_parquet_files(source)) == reference
    assert data_hashes(source, oracle=repo_oracle_cfg())["data_logical_hash"] == reference[0]
    assert (
        training_set_sidecar(source, oracle=repo_oracle_cfg())["data_logical_hash"] == reference[0]
    )
    if pl.__version__ == PINNED_POLARS:
        assert reference[0] == PINNED_C6[subtree]
