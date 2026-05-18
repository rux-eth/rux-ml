"""Solving layer — multi-family Solver registry + ``Solver`` Protocol (per PR-020).

The Solver layer is the parallel-Protocol surface to the Trainer layer
(``rux_ml.training``). Per ``docs/0.1/DESIGN-log.md`` Q1: two parallel
``typing.Protocol``\\ s with no shared parent — ``Trainer.fit(X, y) ->
predict(X)`` and ``Solver.solve(problem) -> SolverResult`` have
fundamentally different data flow; we unify the **factory/registry**
(string-name dispatch via ``make_trainer`` / ``make_solver``) but NOT
the runtime contract.

**Q-Shape (PR-020 Phase 4): B (partial mirror).** The solving layer
reuses cross-cutting workbench infrastructure (config layer, registry
dict, factory dispatcher, Optuna study, per-trial provenance triple,
conformance tests) but bypasses semantics-mismatched layers (``cfg.data``
is tabular; ``cfg.cv`` is fold-based; the registry's two-file model
bundle is fittable-artifact-shaped). Solver runs:

- Have their own ``cfg.solving`` block (Optional on ``RuxMLConfig``).
- Don't have a fittable artifact — output is ``SolverResult.x_star``.
- Don't go through PR-010's promote / champion / bundle path in v0.1.
- DO record per-trial provenance via Optuna ``user_attrs`` (extended
  ``TrialAttrs`` with Optional solver fields).
- Run via ``rux-ml solve`` — a single CLI command (analog of ``rux-ml
  train``), NOT a typer-group; solver-internal HPO is deferred to a
  follow-up PR.

**Q-First-Solver: CVXPY.** Zero dep addition (cvxpy + clarabel already
transitive via skfolio from PR-015; promoted to direct deps in PR-020).
Covers LP/QP/QCQP/SOCP/SDP/MILP through one ``Problem.solve(solver=...)``
switch. Pin ``solver="CLARABEL"`` in TOML for per-trial provenance
stability. Future PRs register clarabel-direct/OSQP-direct as backends
if DCP overhead profiles > 5% of a workbench task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rux_ml.solving.base import SolvingBase
from rux_ml.solving.cvxpy import CVXPYSolving, make_cvxpy_solver
from rux_ml.solving.factory import make_solver
from rux_ml.solving.protocol import Solver
from rux_ml.solving.result import SolverResult

if TYPE_CHECKING:
    from collections.abc import Callable

# B-explicit registry, mirrors ``TRAINER_FAMILIES``. New Solver families
# register themselves here in their landing PR. Order is preserved (Python
# 3.7+ dict insertion order); ``test_registry_conformance.py`` iterates
# sorted-by-key for stable test parametrization.
SOLVER_FAMILIES: dict[str, Callable[..., Solver]] = {
    "cvxpy": make_cvxpy_solver,
}

__all__ = [
    "SOLVER_FAMILIES",
    "CVXPYSolving",
    "Solver",
    "SolverResult",
    "SolvingBase",
    "make_cvxpy_solver",
    "make_solver",
]
