"""Runs layer — Optuna-as-experiment-log read API + provenance recording.

Per D7 / PR-009 plan. PR-007 lands the provenance subset that both the 1-trial
``rux-ml train`` CLI (PR-006) and the sweep ``rux-ml tune`` CLI (PR-007) share.
PR-009 builds query/compare helpers on top.
"""

from rux_ml.runs.provenance import (
    HASH_LAYERS,
    build_user_attrs,
    data_hashes,
    ensure_storage_parent,
    study_name,
)

__all__ = [
    "HASH_LAYERS",
    "build_user_attrs",
    "data_hashes",
    "ensure_storage_parent",
    "study_name",
]
