"""``make_cvxpy_solver`` — CVXPY-specific factory (per PR-020).

Dispatched by ``rux_ml.solving.factory.make_solver`` via the
``SOLVER_FAMILIES["cvxpy"]`` registry entry. Returns a thin solver wrapper
whose ``solve(problem)`` delegates to ``cvxpy.Problem.solve(solver=...)``
and packages the result as a :class:`SolverResult`.

The wrapper captures the workbench-side knobs (solver backend choice,
verbosity, tolerances, max-iter) so the per-trial provenance can hash
them via ``solving_cfg_hash``. Wall-clock solve time is measured at the
Python side (NOT taken from ``problem.solver_stats.solve_time``) so the
recorded time includes cvxpy's DCP compilation + canonicalization
overhead — honest provenance for the workbench.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from rux_ml.solving.result import SolverResult

if TYPE_CHECKING:
    from rux_ml.solving.cvxpy.config import CVXPYSolving
    from rux_ml.solving.protocol import Solver


class _CVXPYSolverWrapper:
    """Thin wrapper around ``cvxpy.Problem.solve()`` that produces a ``SolverResult``.

    Stateless (no fitted parameters between calls; cvxpy's ``Problem``
    holds its own state). Each ``solve(problem)`` call dispatches with
    the configured ``solver=`` + tolerances and returns a fresh
    ``SolverResult``.
    """

    def __init__(self, cfg: CVXPYSolving) -> None:
        self._cfg = cfg

    def solve(self, problem: Any) -> SolverResult:
        """Solve ``problem`` (a ``cvxpy.Problem``) and return a ``SolverResult``.

        The configured ``solver=`` + verbosity + backend-specific
        ``solver_opts`` are passed through. Wall-clock measured at the
        Python boundary so the recorded time includes cvxpy's DCP
        compilation + canonicalization overhead.
        """
        kwargs: dict[str, Any] = {
            "solver": self._cfg.solver,
            "verbose": self._cfg.verbose,
            **self._cfg.solver_opts,
        }

        t_start = time.perf_counter()
        problem.solve(**kwargs)
        t_elapsed = time.perf_counter() - t_start

        return _build_result(problem, t_elapsed)


def _build_result(problem: Any, solve_time_s: float) -> SolverResult:
    """Materialize a :class:`SolverResult` from a solved cvxpy ``Problem``.

    Flattens primal variables in declaration order into a single ``x_star``
    array. Reads ``problem.status`` for the solver status string,
    ``problem.value`` for the objective value (NaN if infeasible /
    unbounded), and ``problem.solver_stats.num_iters`` for the iteration
    count (-1 if not exposed by the backend).
    """
    primals: list[np.ndarray[Any, Any]] = []
    for var in problem.variables():
        v = var.value
        if v is None:
            primals.append(np.array([], dtype=np.float64))
        else:
            arr = np.asarray(v, dtype=np.float64).ravel()
            primals.append(arr)
    x_star = np.concatenate(primals) if primals else np.array([], dtype=np.float64)

    obj_val = problem.value
    objective_value = float(obj_val) if obj_val is not None else math.nan

    status = str(problem.status)

    iter_count = -1
    stats = getattr(problem, "solver_stats", None)
    if stats is not None:
        n = getattr(stats, "num_iters", None)
        if n is not None:
            iter_count = int(n)

    return SolverResult(
        x_star=cast("np.ndarray[Any, Any]", x_star),
        objective_value=objective_value,
        solver_status=status,
        solve_time_s=solve_time_s,
        iter_count=iter_count,
    )


def make_cvxpy_solver(cfg: CVXPYSolving, *, seed: int | None = None) -> Solver:
    """Return the CVXPY solver wrapper for ``cfg``.

    ``seed`` accepted for Trainer-factory signature symmetry but unused
    — cvxpy / clarabel / osqp / scs / ecos / highs are all deterministic
    given inputs. Future stochastic Solver families may use it.
    """
    _ = seed  # reserved for stochastic solver families
    return cast("Solver", _CVXPYSolverWrapper(cfg))
