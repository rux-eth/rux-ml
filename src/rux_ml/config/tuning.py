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
    # Sampler / pruner choice per D6.
    sampler: Literal["tpe", "gp", "botorch", "hebo"] = "tpe"
    pruner: Literal["hyperband", "median", "successive_halving", "wilcoxon", "none"] = "hyperband"

    n_trials: int = 50
    n_startup_trials: int = 20  # for TPE
    multivariate: bool = True  # TPE option
    group: bool = True  # TPE option

    # Subprocess-per-trial isolation per D10/D16 (CUDA + fork is broken).
    trial_isolation: Literal["subprocess", "in_process"] = "subprocess"

    # Optional pinned entropy for reproducibility (per D9). None -> derive per run.
    entropy: int | None = None
