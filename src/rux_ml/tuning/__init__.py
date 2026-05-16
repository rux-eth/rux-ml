"""Tuning layer — Optuna study orchestration + functional objective + factories.

Per D6 + D16 + PR-007 Tier-2 research (locked-in 2026-05-16):

- Sequential trials with TPESampler default (categorical/conditional spaces
  route to TPE per Optuna AutoSampler convention).
- ``WilcoxonPruner`` default for the **K-fold CV-mean** objective (per Optuna
  3.6 design intent for "k-fold cross-validation score of a machine learning
  model"). ``MedianPruner`` is the conservative alternate.
- The objective consumes ``cfg.cv`` via PR-015's ``make_splitter`` and reports
  per-fold scores via ``trial.report(fold_score, fold_idx)`` so the pruner can
  paired-test against the running best trial.
- **No ``XGBoostPruningCallback`` inside the CV loop** (Optuna #3203
  incompatibility); XGBoost-internal ``early_stopping_rounds`` runs per fold.
- Subprocess-per-trial isolation deferred to PR-008.
"""

from rux_ml.tuning.objective import build_objective, walk_search_space
from rux_ml.tuning.pruners import make_pruner
from rux_ml.tuning.samplers import make_sampler
from rux_ml.tuning.study import create_or_load

__all__ = [
    "build_objective",
    "create_or_load",
    "make_pruner",
    "make_sampler",
    "walk_search_space",
]
