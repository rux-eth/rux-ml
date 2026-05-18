"""``Solver`` ``typing.Protocol`` — the universal cross-family solver surface.

Per ``docs/0.1/DESIGN-log.md`` Q1: ``Trainer.fit(X, y) -> predict(X)`` and
``Solver.solve(problem) -> SolverResult`` have fundamentally different
data flow. The Solver Protocol intentionally covers only the universal
subset (``solve``). The ``problem`` type is ``Any`` because future Solver
families take different problem types — cvxpy uses ``cvxpy.Problem``;
OSQP-direct uses raw matrix tuples; pyomo-direct uses ``ConcreteModel``.

Like the Trainer Protocol (per PR-017's conformance-test research), the
Solver Protocol is NOT ``@runtime_checkable``: PEP 544 + CPython 3.12
typing docs note that runtime isinstance checks only verify method names,
not signatures. The behavioral conformance test
(``tests/solving/test_registry_conformance.py``) is the runtime gate —
it parametrizes over ``SOLVER_FAMILIES`` and asserts each factory returns
an object whose ``solve()`` produces a valid ``SolverResult``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rux_ml.solving.result import SolverResult


class Solver(Protocol):
    """Universal cross-family solver surface (``solve(problem) -> SolverResult``)."""

    def solve(self, problem: Any) -> SolverResult: ...
