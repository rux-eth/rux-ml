"""``Trainer`` ``typing.Protocol`` — the universal-subset of the sklearn estimator API.

Per D5 (``docs/0.0/DESIGN-log.md``): every concrete model family in this workbench
exposes the sklearn estimator API, and the workbench treats that surface as the
unifying contract. The Protocol intentionally covers only the **universal
subset** (``fit`` + ``predict``) — fields that differ between classifiers and
regressors live outside it:

- ``predict_proba`` — classification-only; the metric registry narrows via
  ``cast`` at the call sites for AUC / logloss.
- ``best_iteration_`` — populated by XGBoost only after a fit with
  ``early_stopping_rounds`` set; the CLI accesses it via ``getattr(..., None)``
  so the absence (without early stopping) is not a runtime failure.

Keeping the Protocol minimal lets both ``XGBClassifier`` and ``XGBRegressor``
satisfy it structurally per the v0 stubs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class Trainer(Protocol):
    """Universal cross-family trainer surface (``fit`` + ``predict``)."""

    def fit(self, X: Any, y: ArrayLike, **kwargs: Any) -> Trainer: ...

    def predict(self, X: Any) -> ArrayLike: ...
