"""Optuna study create-or-load wrapper.

Thin layer over ``optuna.create_study`` that:
- ensures the SQLite parent directory exists (via ``runs.provenance.ensure_storage_parent``);
- defaults to ``load_if_exists=True`` so ``rux-ml tune resume`` is idempotent;
- passes the configured sampler + pruner + direction (per the metric).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optuna

from rux_ml.runs import ensure_storage_parent

if TYPE_CHECKING:
    from optuna.pruners import BasePruner
    from optuna.samplers import BaseSampler
    from optuna.study import Study


def create_or_load(
    name: str,
    storage: str,
    sampler: BaseSampler,
    pruner: BasePruner,
    direction: str,
    *,
    load_if_exists: bool = True,
) -> Study:
    """Create or load an Optuna study with the given storage + sampler + pruner.

    Args:
        name: Study identifier.
        storage: Optuna storage URL (e.g. ``sqlite:///studies/studies.db``).
            Parent directory is created if missing.
        sampler: Constructed via ``rux_ml.tuning.make_sampler``.
        pruner: Constructed via ``rux_ml.tuning.make_pruner``.
        direction: ``"maximize"`` / ``"minimize"`` per the metric.
        load_if_exists: Idempotent load (default True). Set False to require a
            fresh study and surface a clear error on collision.
    """
    ensure_storage_parent(storage)
    return optuna.create_study(
        study_name=name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=load_if_exists,
    )
