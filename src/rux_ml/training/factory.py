"""``make_trainer`` top-level dispatcher (per PR-017).

Replaces the pre-PR-017 single-family if/elif dispatch with a registry-driven
lookup against ``TRAINER_FAMILIES`` (declared in
:mod:`rux_ml.training.__init__`). The signature is preserved — every existing
caller (``cli/train.py``, ``tuning/objective.py``, ``registry/promote.py``,
tests) continues to call ``make_trainer(cfg.training, seed=...)`` unchanged.

Per Q4 of ``docs/0.1/DESIGN-log.md``: the registry is a plain dict, not a
decorator-driven plugin loader. Each family's factory is registered explicitly
at module-import time when ``TRAINER_FAMILIES`` is constructed. The
parametrized conformance test (``tests/training/test_registry_conformance.py``)
iterates the dict so any registered family that fails to satisfy the
:class:`Trainer` Protocol fails CI loudly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rux_ml.training.xgboost.config import XGBoostTraining

if TYPE_CHECKING:
    from rux_ml.config.training import TrainingConfig
    from rux_ml.training.protocol import Trainer


def make_trainer(cfg: TrainingConfig, *, seed: int | None = None) -> Trainer:
    """Return the concrete trainer for ``cfg``.

    Dispatches on ``cfg.kind`` via the ``TRAINER_FAMILIES`` registry. Each
    registered family is responsible for narrowing ``cfg`` to its own
    variant type (e.g. ``XGBoostTraining``) and constructing the concrete
    estimator.

    ``seed`` (PR-013) is passed through to the family factory; the family
    decides where to plumb it (XGBoost uses ``random_state``).

    Raises:
        ValueError: when ``cfg.kind`` is not in ``TRAINER_FAMILIES``. This
            happens when a config selects a family that isn't registered —
            e.g. a stale TOML referencing a removed family, or a family
            scoped to a future PR that hasn't landed yet.
    """
    from rux_ml.training import TRAINER_FAMILIES  # noqa: PLC0415 — break import cycle

    family = cfg.kind
    factory = TRAINER_FAMILIES.get(family)
    if factory is None:
        known = sorted(TRAINER_FAMILIES)
        msg = (
            f"training.kind={family!r} is not registered in TRAINER_FAMILIES; "
            f"known families: {known}. New families land in their own follow-up "
            f"PRs and must register themselves in src/rux_ml/training/__init__.py."
        )
        raise ValueError(msg)
    return factory(_narrow_for_family(cfg, family), seed=seed)


def _narrow_for_family(cfg: TrainingConfig, family: str) -> TrainingConfig:
    """Narrow ``cfg`` to the registered variant type for ``family``.

    Pydantic v2's discriminator handles this at validation time — by the time
    ``make_trainer`` is called, ``cfg`` is already the correct variant subclass.
    This helper is a no-op at runtime but documents the narrowing intent for
    future readers and gives us a place to harden the invariant if a
    non-Pydantic caller ever bypasses validation.
    """
    # ``isinstance`` is trivially-true when the union has a single variant
    # (TrainingConfig = Annotated[XGBoostTraining, ...]), but becomes
    # meaningful in PR-018 once LightGBMTraining widens the union — this
    # narrows ``cfg`` correctly for downstream factories.
    if family == "xgboost" and not isinstance(cfg, XGBoostTraining):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = (
            f"cfg.kind=='xgboost' but cfg is not an XGBoostTraining instance "
            f"({type(cfg).__name__}); did a caller bypass Pydantic validation?"
        )
        raise TypeError(msg)
    return cfg
