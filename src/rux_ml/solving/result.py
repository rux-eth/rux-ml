"""``SolverResult`` dataclass — the runtime output of every Solver family.

Plain ``@dataclass`` (not Pydantic): this is a runtime result type, not
a config. Designed for direct consumption by the CLI's ``rux-ml solve``
command and by future per-family smoke / regression tests. Fields chosen
to capture the minimal common surface across cvxpy backends (Clarabel,
OSQP, SCS, ECOS, HiGHS) so per-trial provenance can record them
uniformly via :class:`rux_ml.runs.TrialAttrs`.

Fields are loosely modeled after cvxpy's `Problem.solver_stats` +
`Problem.status` + `Problem.value` surface, plus the workbench's
provenance-triple convention (numeric metrics + status string).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SolverResult:
    """Universal cross-family solver result."""

    # The optimal primal solution. Shape depends on the problem; for cvxpy
    # multi-variable problems this is the flattened concatenation of all
    # variables in declaration order (matches cvxpy's ``problem.solution.primal_vars``
    # serialization but as a single array). For single-variable problems it
    # is the variable's value.
    x_star: NDArray[Any]

    # Optimal objective value. NaN if the problem was infeasible / unbounded.
    objective_value: float

    # Solver-reported status string. Common cvxpy values: "optimal",
    # "optimal_inaccurate", "infeasible", "unbounded", "user_limit",
    # "solver_error". The workbench treats anything other than "optimal" /
    # "optimal_inaccurate" as a non-converged trial.
    solver_status: str

    # Wall-clock solve time in seconds (measured by the factory, NOT the
    # solver — captures Python-side dispatch overhead for honest provenance).
    solve_time_s: float

    # Iteration count reported by the solver. -1 when not available (some
    # backends don't expose it).
    iter_count: int
