"""Solving layer config — discriminated union over per-family Solver variants (per PR-020).

Re-exports :class:`SolvingBase` (from ``rux_ml.solving.base``) and the
CVXPY variant :class:`CVXPYSolving` (from ``rux_ml.solving.cvxpy.config``)
so existing callers' imports (``from rux_ml.config import SolvingConfig``)
resolve. The discriminated-union :data:`SolvingConfig` is assembled here
at the config-layer level — mirrors the v0.1 pattern in
``rux_ml/config/training.py``.

PR-020 ships with a single CVXPY variant; future Solver-family PRs widen
the union by adding new variants in the ``Annotated[...]`` below.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from rux_ml.solving.base import SolvingBase
from rux_ml.solving.cvxpy.config import CVXPYSolving

# Discriminated-union SolvingConfig. Single variant in v0.1; future PRs
# widen the Union as new Solver families land.
SolvingConfig = Annotated[
    CVXPYSolving,
    Field(discriminator="kind"),
]

__all__ = [
    "CVXPYSolving",
    "SolvingBase",
    "SolvingConfig",
]
