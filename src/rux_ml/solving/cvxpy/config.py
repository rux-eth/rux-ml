"""``CVXPYSolving`` — discriminated-union variant for CVXPY (per PR-020).

Inherits ``problem_module`` from :class:`rux_ml.solving.base.SolvingBase`.
The family-discriminator ``kind`` is a CVXPY-specific Literal. Per the
PR-020 Q-First-Solver research, ``solver`` is pinned to one of cvxpy's
registered backends (Clarabel default for QP/SOCP/SDP; OSQP for pure QP;
SCS for conic; ECOS for legacy; HiGHS for LP/MILP).

Per-trial provenance: ``solving_cfg_hash`` is computed via the standard
``layer_cfg_hash`` machinery; ``problem_module`` and ``solver`` both
contribute. (See ``docs/CONVENTIONS.md`` solver-side conventions for the
known limitation that the module's source content is NOT hashed.)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rux_ml.solving.base import SolvingBase


class CVXPYSolving(SolvingBase):
    """CVXPY variant of the discriminated ``SolvingConfig``."""

    kind: Literal["cvxpy"] = "cvxpy"

    # Which cvxpy backend to dispatch to. Clarabel is the recommended
    # default for QP/SOCP/SDP (modern Rust solver; ranks #3 in
    # qpsolvers/free_for_all_qpbenchmark per PR-020 Q-First-Solver
    # research). OSQP for pure QP fast paths; SCS for conic; ECOS as a
    # mature legacy fallback; HIGHS for LP / MILP.
    solver: Literal["CLARABEL", "OSQP", "SCS", "ECOS", "HIGHS"] = "CLARABEL"

    # Verbosity flag for the underlying solver. Default False — the
    # workbench owns user-facing logging.
    verbose: bool = False

    # Backend-specific settings escape hatch. Each cvxpy backend has its
    # own tolerance / max-iter / refinement knobs with DIFFERENT names —
    # Clarabel uses ``tol_gap_abs`` / ``tol_feas`` / ``max_iter``; OSQP
    # uses ``eps_abs`` / ``eps_rel`` / ``max_iter``; SCS uses different
    # names again. Rather than fake a unified abstraction across backend
    # names (which would be a leaky abstraction), the workbench passes
    # whatever's in ``solver_opts`` straight through to ``Problem.solve(
    # solver=..., **solver_opts)``. Users consult the backend's docs for
    # the right names. (Matches the Trainer-side ``model_kwargs`` escape
    # hatch on XGBoostTraining / LightGBMTraining.)
    solver_opts: dict[str, Any] = Field(default_factory=dict)
