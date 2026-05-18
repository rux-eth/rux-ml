"""``make_solver`` top-level dispatcher (per PR-020 / Q4 of docs/0.1/DESIGN-log.md).

Mirrors :func:`rux_ml.training.factory.make_trainer`. Dispatches on
``cfg.solving.kind`` against ``SOLVER_FAMILIES`` (declared in
:mod:`rux_ml.solving.__init__`). Every existing caller passes
``make_solver(cfg.solving, *, seed=None)`` unchanged across families;
the dispatcher routes to the family factory which knows how to narrow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rux_ml.solving.cvxpy.config import CVXPYSolving

if TYPE_CHECKING:
    from rux_ml.config.solving import SolvingConfig
    from rux_ml.solving.protocol import Solver


def make_solver(cfg: SolvingConfig, *, seed: int | None = None) -> Solver:
    """Return the concrete Solver for ``cfg``.

    Dispatches on ``cfg.kind`` via the ``SOLVER_FAMILIES`` registry. Each
    family's factory narrows ``cfg`` to its own variant type and returns
    an object satisfying the :class:`Solver` Protocol.

    ``seed`` (PR-013 signature symmetry with the Trainer factory): not
    used by cvxpy / clarabel (deterministic given inputs) but accepted
    so future stochastic Solver families slot in without changing the
    dispatcher signature.

    Raises:
        ValueError: when ``cfg.kind`` is not in ``SOLVER_FAMILIES``.
    """
    from rux_ml.solving import SOLVER_FAMILIES  # noqa: PLC0415 — break import cycle

    family = cfg.kind  # pyright: ignore[reportAttributeAccessIssue]
    factory = SOLVER_FAMILIES.get(family)
    if factory is None:
        known = sorted(SOLVER_FAMILIES)
        msg = (
            f"solving.kind={family!r} is not registered in SOLVER_FAMILIES; "
            f"known families: {known}. New families register themselves in "
            f"src/rux_ml/solving/__init__.py."
        )
        raise ValueError(msg)
    return factory(_narrow_for_family(cfg, family), seed=seed)  # pyright: ignore[reportCallIssue]


def _narrow_for_family(cfg: SolvingConfig, family: str) -> SolvingConfig:
    """Narrow ``cfg`` to the registered variant type for ``family``.

    Pydantic v2's discriminator routes the input dict to the matching
    variant at validation time; by the time ``make_solver`` runs, ``cfg``
    is already the correct subclass. The ``isinstance`` check is
    belt-and-suspenders for callers that bypass Pydantic validation.
    """
    # ``isinstance`` is trivially-true when the union has a single variant
    # (SolvingConfig = Annotated[CVXPYSolving, ...]) but becomes meaningful
    # when future Solver families widen the union.
    if family == "cvxpy" and not isinstance(cfg, CVXPYSolving):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = (
            f"cfg.kind=='cvxpy' but cfg is not a CVXPYSolving instance "
            f"({type(cfg).__name__}); did a caller bypass Pydantic validation?"
        )
        raise TypeError(msg)
    return cfg
