"""Family-agnostic ``SolvingBase`` (per PR-020 / docs/0.1/DESIGN-log.md Q1).

Mirrors :class:`rux_ml.training.base.TrainingBase` — a neutral Pydantic
model with strict config that per-family Solver variants inherit. Lives
outside ``rux_ml.config.*`` to avoid the circular import via
``rux_ml.config.__init__`` (same pattern as ``training.base``).

Fields here are universal across Solver families: a Python import path
to the problem-builder module, and the family-discriminator (declared
on each variant as ``kind: Literal[<family>]``, not on the base, to
avoid the ``reportIncompatibleVariableOverride`` trap per PR-017's
research).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SolvingBase(BaseModel):
    """Family-agnostic solving-config fields shared across all Solver families."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    )

    # Python import path to a module that exports ``build_problem() ->
    # <family-specific problem type>``. For cvxpy: the module returns a
    # ``cvxpy.Problem``. The CLI (``rux-ml solve``) imports this module and
    # calls ``build_problem()`` before passing the result to
    # ``solver.solve(problem)``.
    #
    # Hashing limitation: ``solving_cfg_hash`` captures this path string
    # but NOT the module's source content. Keep problem modules inside the
    # workbench's git so ``git_sha`` covers edits (per
    # ``docs/CONVENTIONS.md`` solver-side conventions).
    problem_module: str
