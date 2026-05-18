"""Parametrized Protocol-conformance test for ``SOLVER_FAMILIES`` (per PR-020).

Mirrors :mod:`tests.training.test_registry_conformance` for the solving
layer. Iterates every registered Solver family in
:data:`rux_ml.solving.SOLVER_FAMILIES` and asserts:

1. The factory returns a non-None object.
2. The returned object exposes ``solve`` as a callable.
3. End-to-end ``solve(problem) -> SolverResult`` smoke succeeds on a
   family-specific minimal problem, with the result reporting a valid
   ``solver_status`` (optimal or optimal_inaccurate) and finite
   ``objective_value`` + ``x_star``.

This is the stale-path tripwire for the Solver registry. Registering a
family in ``src/rux_ml/solving/__init__.py`` without a working subpackage
fails this test; removing a family by deleting its subpackage but
forgetting to remove the registry entry fails this test.

Per PR-020 Q-First-Solver: the first registered family is ``cvxpy``;
future families add entries to ``_MINIMAL_CFG`` + ``_MINIMAL_PROBLEM``
with their own problem types (raw matrices for OSQP-direct, etc).
"""

from __future__ import annotations

import math
from typing import Any

import cvxpy as cp
import numpy as np
import pytest

from rux_ml.solving import SOLVER_FAMILIES, CVXPYSolving, SolverResult, SolvingBase


def _build_cvxpy_min_problem() -> cp.Problem:
    """Tiny 2-variable convex QP for the CVXPY conformance smoke."""
    x = cp.Variable(2)
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([1.0, 1.0])
    obj = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x)
    return cp.Problem(obj, [x >= 0])


_MINIMAL_CFG: dict[str, SolvingBase] = {
    "cvxpy": CVXPYSolving(problem_module="<conformance-test-inline>", solver="CLARABEL"),
}

_MINIMAL_PROBLEM: dict[str, Any] = {
    "cvxpy": _build_cvxpy_min_problem(),
}


def test_minimal_cfg_covers_every_registered_family() -> None:
    """Fail loudly if a new family lands in ``SOLVER_FAMILIES`` without a
    corresponding minimal-config entry here. The conformance test below would
    KeyError otherwise, which is less informative than a dedicated assertion.
    """
    missing_cfg = set(SOLVER_FAMILIES) - set(_MINIMAL_CFG)
    missing_problem = set(SOLVER_FAMILIES) - set(_MINIMAL_PROBLEM)
    assert not missing_cfg, (
        f"Family added to SOLVER_FAMILIES but missing from _MINIMAL_CFG: {missing_cfg}. "
        f"Add a minimal-config entry for each new family in test_registry_conformance.py."
    )
    assert not missing_problem, (
        f"Family added to SOLVER_FAMILIES but missing from _MINIMAL_PROBLEM: "
        f"{missing_problem}. Add a minimal-problem entry too."
    )


@pytest.mark.parametrize("family", sorted(SOLVER_FAMILIES))
def test_family_conforms_to_solver_protocol(family: str) -> None:
    """Every registered Solver family satisfies the behavioral ``Solver`` contract.

    (a) factory returns non-None
    (b) ``solve`` is callable
    (c) end-to-end ``solve(problem)`` smoke succeeds
    (d) result is a :class:`SolverResult` with optimal-or-better status
    (e) ``x_star`` is finite
    """
    factory = SOLVER_FAMILIES[family]
    cfg = _MINIMAL_CFG[family]
    problem = _MINIMAL_PROBLEM[family]

    # (a) factory returns non-None
    solver = factory(cfg)
    assert solver is not None

    # (b) required method callable
    assert callable(getattr(solver, "solve", None)), f"{family}: missing callable 'solve'"

    # (c) end-to-end solve
    result = solver.solve(problem)

    # (d) returns a SolverResult
    assert isinstance(result, SolverResult), f"{family}: solve did not return a SolverResult"
    assert result.solver_status in {"optimal", "optimal_inaccurate"}, (
        f"{family}: solver_status={result.solver_status!r} (expected optimal / optimal_inaccurate)"
    )

    # (e) x_star finite + objective_value finite
    assert np.isfinite(result.x_star).all(), f"{family}: x_star contains non-finite values"
    assert math.isfinite(result.objective_value), (
        f"{family}: objective_value={result.objective_value} is not finite"
    )
    assert result.solve_time_s >= 0.0, f"{family}: solve_time_s={result.solve_time_s} is negative"
