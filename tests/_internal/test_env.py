"""Tests for ``rux_ml._internal.env.pin_threads`` (per PR-011 sub-decision A1)."""

from __future__ import annotations

import os

import pytest

from rux_ml._internal.env import pin_threads
from rux_ml.config import MemoryConfig

_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS")


def test_pin_threads_sets_all_four_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    pin_threads(MemoryConfig(omp_threads=24, openblas_threads=1, mkl_threads=1, polars_threads=24))
    assert os.environ["OMP_NUM_THREADS"] == "24"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["POLARS_MAX_THREADS"] == "24"


def test_pin_threads_overwrites_inherited_shell_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inherited shell values must not win — workbench config is authoritative."""
    for v in _VARS:
        monkeypatch.setenv(v, "999")
    pin_threads(MemoryConfig(omp_threads=8, openblas_threads=2, mkl_threads=3, polars_threads=4))
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert os.environ["POLARS_MAX_THREADS"] == "4"


def test_pin_threads_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    memory = MemoryConfig(omp_threads=24, openblas_threads=1, mkl_threads=1, polars_threads=24)
    pin_threads(memory)
    pin_threads(memory)
    assert os.environ["OMP_NUM_THREADS"] == "24"
