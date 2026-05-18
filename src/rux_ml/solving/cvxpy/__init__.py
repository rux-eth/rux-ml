"""CVXPY solver family — DCP modeling layer over Clarabel / OSQP / SCS / ECOS / HiGHS.

Per PR-020 Q-First-Solver research: ships as the first registered Solver
because (a) zero dep addition (cvxpy + clarabel already transitive via
skfolio, promoted to direct deps in PR-020), (b) broad problem-class
coverage (LP, QP, QCQP, SOCP, SDP, MILP), (c) Python ergonomics (DCP
syntax reads like the math), and (d) one ``solver=`` switch unlocks every
registered backend.

Pin ``solver="CLARABEL"`` in TOML for per-trial provenance stability
(Clarabel ranks #3 in qpsolvers/free_for_all_qpbenchmark; modern Rust
solver; Apache-2.0).

Future PRs register clarabel-direct / OSQP-direct as backends if DCP
overhead profiles > 5% of a workbench task (analog of the v0 "first Rust
crate" threshold from D13).
"""

from rux_ml.solving.cvxpy.config import CVXPYSolving
from rux_ml.solving.cvxpy.factory import make_cvxpy_solver

__all__ = [
    "CVXPYSolving",
    "make_cvxpy_solver",
]
