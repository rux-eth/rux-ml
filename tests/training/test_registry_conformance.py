"""Parametrized Protocol-conformance test for ``TRAINER_FAMILIES`` (per PR-017).

Iterates every registered Trainer family in ``TRAINER_FAMILIES`` and asserts:

1. The factory returns a non-None object.
2. The returned object exposes ``fit`` and ``predict`` as callables (mirroring
   ``@runtime_checkable``'s name-presence check, which we intentionally do NOT
   add to the ``Trainer`` Protocol itself — see ``docs/0.1/DESIGN-log.md`` Q-CT
   for the PEP 544 / 3.12-typing-docs rationale).
3. End-to-end ``fit(X, y) → predict(X)`` smoke succeeds on a tiny synthetic
   dataset, with ``fit`` returning the trainer (sklearn convention) and
   ``predict`` returning an array of the expected shape, finite values only.

This is the stale-path tripwire that makes the B-explicit registry pattern
safe: registering a family in ``src/rux_ml/training/__init__.py`` without a
working subpackage fails this test. Removing a family by deleting its
subpackage but forgetting to remove its registry entry fails this test.

Anchored on sklearn's ``parametrize_with_checks`` battery
(``sklearn/utils/estimator_checks.py``) and Optuna's
``pytest_samplers.py`` shared parametrized suite — both reused across
every built-in implementation of their respective protocol surfaces.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from rux_ml.training import TRAINER_FAMILIES, TrainingBase, XGBoostTraining

# Minimal per-family configs for the conformance smoke. Each entry constructs
# a config valid for that family on CPU (so GPU-free CI machines pass the test
# unconditionally) with classification metric + early stopping disabled.
#
# New families register their minimal config here in their landing PR. The
# fail-loudly assertion below catches the case where a family is added to
# ``TRAINER_FAMILIES`` but not here.
_MINIMAL_CFG: dict[str, TrainingBase] = {
    "xgboost": XGBoostTraining(
        device="cpu",
        metric="auc",
        n_estimators=8,
        max_depth=3,
        learning_rate=0.3,
        early_stopping_rounds=None,
    ),
}


def test_minimal_cfg_covers_every_registered_family() -> None:
    """Fail loudly if a new family lands in ``TRAINER_FAMILIES`` without a
    corresponding minimal-config entry here. The conformance test below would
    KeyError otherwise, which is less informative than a dedicated assertion.
    """
    missing = set(TRAINER_FAMILIES) - set(_MINIMAL_CFG)
    assert not missing, (
        f"Family added to TRAINER_FAMILIES but missing from _MINIMAL_CFG: {missing}. "
        f"Add a minimal-config entry for each new family in test_registry_conformance.py."
    )


@pytest.mark.parametrize("family", sorted(TRAINER_FAMILIES))
def test_family_conforms_to_trainer_protocol(family: str) -> None:
    """Every registered Trainer family satisfies the behavioral ``Trainer`` contract.

    (a) factory returns non-None
    (b) ``fit`` and ``predict`` are callable
    (c) ``fit(X, y)`` succeeds on ``make_classification(50, 4)``
    (d) ``fit`` returns the trainer (sklearn fit-returns-self convention)
    (e) ``predict(X)`` returns a finite array of shape ``(n_samples,)``
    """
    factory = TRAINER_FAMILIES[family]
    cfg = _MINIMAL_CFG[family]

    # (a) factory returns non-None
    trainer = factory(cfg)
    assert trainer is not None

    # (b) required methods present and callable
    assert callable(getattr(trainer, "fit", None)), f"{family}: missing callable 'fit'"
    assert callable(getattr(trainer, "predict", None)), f"{family}: missing callable 'predict'"

    # (c) end-to-end fit-predict smoke
    x, y = make_classification(n_samples=50, n_features=4, random_state=0)
    fitted = trainer.fit(x, y)

    # (d) fit returns self
    assert fitted is trainer, f"{family}: fit did not return self (sklearn convention)"

    # (e) predict returns finite array of correct shape
    yhat = np.asarray(trainer.predict(x))
    assert yhat.shape == (50,), f"{family}: predict shape {yhat.shape} != (50,)"
    assert np.isfinite(yhat).all(), f"{family}: predict returned non-finite values"
