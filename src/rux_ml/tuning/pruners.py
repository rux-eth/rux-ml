"""Pruner factory (per PR-007 Tier-2 research, sub-decision A1).

Locked-in default:
- ``"wilcoxon"`` (default) — ``WilcoxonPruner``. Purpose-built for K-fold
  CV-mean objectives in Optuna 3.6+: per-fold scores are reported via
  ``trial.report(fold_score, fold_idx)`` and the pruner runs a paired Wilcoxon
  signed-rank test against the running best trial.
- ``"median"`` — ``MedianPruner``. Conservative alternate; production-grade
  (Optuna's own ``xgboost_cv_integration.py`` example uses it). Loses the
  per-fold statistical guarantee but ships a stable non-experimental API.

The remaining literal values (``"hyperband"``, ``"successive_halving"``,
``"none"``) are preserved for hypothetical single-fit objectives in future
PRs — they're iteration-level pruners that don't match the K-fold CV regime
PR-007 lands.

WilcoxonPruner's "TPESampler currently cannot utilize the information of
pruned trials effectively" caveat (per Optuna's own docs) is the documented
trade-off; users seeking that interaction can switch to MedianPruner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    SuccessiveHalvingPruner,
    WilcoxonPruner,
)

if TYPE_CHECKING:
    from rux_ml.config import TuningConfig

_WILCOXON_P_THRESHOLD = 0.1  # Optuna's WilcoxonPruner default; tunable in a follow-up if needed.
_WILCOXON_N_STARTUP_STEPS = 2  # Low-K safety: require at least 2 folds before pruning kicks in.


def make_pruner(cfg: TuningConfig) -> BasePruner:
    """Return the configured Optuna pruner."""
    kind = cfg.pruner
    if kind == "wilcoxon":
        return WilcoxonPruner(
            p_threshold=_WILCOXON_P_THRESHOLD,
            n_startup_steps=_WILCOXON_N_STARTUP_STEPS,
        )
    if kind == "median":
        return MedianPruner(n_startup_trials=cfg.n_startup_trials)
    if kind == "hyperband":
        return HyperbandPruner()
    if kind == "successive_halving":
        return SuccessiveHalvingPruner()
    if kind == "none":
        return NopPruner()
    msg = f"unknown pruner kind {kind!r}"  # pragma: no cover — Literal type narrows this
    raise ValueError(msg)
