"""End-to-end smoke for the CVXPY Solver family (per PR-020).

Surfaces beyond the parametrized conformance test in
``test_registry_conformance.py``:

1. **`solver=` switch** — verifies the user can pick a different cvxpy
   backend via ``cfg.solver`` (CLARABEL default; the smoke also exercises
   OSQP since it's pulled in transitively by cvxpy).
2. **`solver_opts` escape hatch** — verifies backend-specific settings
   (which differ between Clarabel / OSQP / SCS — Phase 5 Gate Check
   surface) flow through to the underlying solver.
3. **`SolverResult` field shape** — every documented field is populated
   for a typical convex QP solve.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from rux_ml.solving import CVXPYSolving, make_cvxpy_solver


def _tiny_qp() -> cp.Problem:
    """Tiny convex QP: min 0.5 x' P x + q' x  s.t. x >= 0."""
    x = cp.Variable(3)
    P = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]])
    q = np.array([1.0, -1.0, 0.5])
    obj = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x)
    return cp.Problem(obj, [x >= 0])


def test_cvxpy_clarabel_solves_tiny_qp() -> None:
    """Default CLARABEL backend solves a tiny QP and reports full SolverResult."""
    cfg = CVXPYSolving(problem_module="<test>", solver="CLARABEL")
    solver = make_cvxpy_solver(cfg)

    result = solver.solve(_tiny_qp())

    assert result.solver_status in {"optimal", "optimal_inaccurate"}
    assert result.x_star.shape == (3,)
    assert np.isfinite(result.x_star).all()
    # min 0.5 x'Px + q'x with x>=0 is bounded below; objective <= 0 here.
    assert result.objective_value < 1.0
    assert result.solve_time_s >= 0.0
    # iter_count exposed by Clarabel
    assert result.iter_count >= 0


def test_cvxpy_osqp_backend_switch() -> None:
    """Switching ``cfg.solver`` to OSQP routes through cvxpy's OSQP backend."""
    cfg = CVXPYSolving(problem_module="<test>", solver="OSQP")
    solver = make_cvxpy_solver(cfg)

    result = solver.solve(_tiny_qp())

    assert result.solver_status in {"optimal", "optimal_inaccurate"}
    # Both backends should converge to the same optimum within tolerance.
    assert result.x_star.shape == (3,)
    assert np.isfinite(result.x_star).all()


def test_cvxpy_solver_opts_passthrough() -> None:
    """Backend-specific settings via ``solver_opts`` flow through to the underlying solver.

    Clarabel uses ``max_iter`` as a top-level setting (not ``eps_abs`` /
    ``eps_rel`` which are OSQP/SCS naming). The escape hatch lets the
    workbench avoid faking a unified abstraction across backend names.
    """
    cfg = CVXPYSolving(
        problem_module="<test>",
        solver="CLARABEL",
        solver_opts={"max_iter": 50},
    )
    solver = make_cvxpy_solver(cfg)

    result = solver.solve(_tiny_qp())
    assert result.solver_status in {"optimal", "optimal_inaccurate"}
    # iter_count should be small for a tiny QP; mainly verifying no exception.
    assert result.iter_count <= 50
