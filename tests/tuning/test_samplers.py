"""Tests for the sampler factory (sub-decisions B1 + C1)."""

from __future__ import annotations

from unittest.mock import patch

import optuna
import pytest
from optuna.samplers import GPSampler, TPESampler

from rux_ml.config import TuningConfig
from rux_ml.tuning import make_sampler


def test_make_sampler_returns_tpesampler_with_constant_liar() -> None:
    cfg = TuningConfig(sampler="tpe", constant_liar=True)
    sampler = make_sampler(cfg, seed=0)
    assert isinstance(sampler, TPESampler)


def test_make_sampler_returns_gpsampler() -> None:
    cfg = TuningConfig(sampler="gp")
    sampler = make_sampler(cfg, seed=0)
    assert isinstance(sampler, GPSampler)


def test_make_sampler_hebo_without_optunahub_raises_clear_import_error() -> None:
    cfg = TuningConfig(sampler="hebo")
    # Patch the lazy import inside _make_hebo to simulate optunahub being absent.
    with (
        patch.dict("sys.modules", {"optunahub": None}),
        pytest.raises(ImportError, match="pip install optunahub hebo"),
    ):
        make_sampler(cfg, seed=0)


def test_make_sampler_threads_seed_into_tpe() -> None:
    """Same seed produces the same first-trial suggestion (determinism)."""
    cfg = TuningConfig(sampler="tpe", n_startup_trials=1)

    def study_with(seed: int) -> float:
        study = optuna.create_study(sampler=make_sampler(cfg, seed=seed))
        study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
        return study.trials[0].params["x"]

    assert study_with(7) == study_with(7)
    assert study_with(7) != study_with(8)
