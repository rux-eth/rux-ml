"""Tuning layer config + SearchSpec tagged-union (per D6, D10, D16)."""

from typing import Annotated, Literal

from pydantic import Field

from rux_ml.config._strict_model import StrictModel

# ---------- SearchSpec tagged union (per D16) ----------


class FloatSpec(StrictModel):
    type: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False


class IntSpec(StrictModel):
    type: Literal["int"] = "int"
    low: int
    high: int
    log: bool = False


class CatSpec(StrictModel):
    type: Literal["categorical"] = "categorical"
    choices: list[str | int | float]


SearchSpec = Annotated[
    FloatSpec | IntSpec | CatSpec,
    Field(discriminator="type"),
]


# ---------- TuningConfig ----------


class TuningConfig(StrictModel):
    # Sampler choice — narrowed by PR-007 Tier-2 research (2026-05-16). BoTorchSampler
    # was deprecated for single-objective HPO in Optuna 3.6 (~5x slower than GPSampler
    # with no cited tabular-GBM advantage) and is no longer offered. HEBO is opt-in
    # via `optunahub` — the sampler factory raises a clear ImportError pointing at
    # `pip install optunahub hebo` when selected without the optional deps.
    sampler: Literal["tpe", "gp", "hebo"] = "tpe"

    # Pruner choice — PR-007 research locked WilcoxonPruner as default for the
    # K-fold CV-mean objective (per-fold statistical test). MedianPruner is the
    # conservative alternate (production-grade — Optuna's own xgboost_cv_integration
    # example uses it). Hyperband / SuccessiveHalving preserved for hypothetical
    # future single-fit objectives.
    pruner: Literal["hyperband", "median", "successive_halving", "wilcoxon", "none"] = "wilcoxon"

    n_trials: int = 50
    n_startup_trials: int = 20  # for TPE
    multivariate: bool = True  # TPE option
    group: bool = True  # TPE option
    constant_liar: bool = True  # TPE option — matches Optuna AutoSampler's TPE config

    # Subprocess-per-trial isolation per D10/D16 (CUDA + fork is broken).
    trial_isolation: Literal["subprocess", "in_process"] = "subprocess"

    # Per-trial wall-clock timeout (seconds) for the subprocess path (PR-008).
    # None disables the timeout. The parent kills a hanging child after
    # ``trial_timeout_s`` and marks the trial as FAIL in storage.
    trial_timeout_s: int | None = None

    # Optional pinned entropy for reproducibility (per D9). None -> derive per run.
    entropy: int | None = None
