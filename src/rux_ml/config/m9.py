"""The optional ``[m9]`` layer (PR-041; rux-capital program v0.3 D37 / D45 #3).

An M9 problem trains on exactly one label — the column the program's EV consumes
(``data.target_column``: ``y__fill_frac``, ``y__markout_bp`` at the decision
interval, or ``y__walk_bp``). The other labels of the same training table are
**diagnostics**: summarised beside every fold, never trained on and never among
the features. ``RuxMLConfig`` enforces both rules at config load.

The layer is optional so every non-M9 config keeps its recorded hashes: an
unset optional layer contributes nothing to ``root_cfg_hash`` (``config/root.py``
``_HASH_OPTIONAL_LAYERS``); once set it is part of the trial's identity.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from rux_ml.config._strict_model import StrictModel


class M9Config(StrictModel):
    """Per-problem M9 settings. Later PRs of the R-14 track add ``honesty`` and
    ``learning_curve`` sub-tables here."""

    # Labels carried beside the target as diagnostics (distinct, non-empty names).
    diagnostic_columns: list[str] = Field(default_factory=list)

    @field_validator("diagnostic_columns")
    @classmethod
    def _distinct_non_empty(cls, v: list[str]) -> list[str]:
        if any(not c for c in v):
            msg = "m9.diagnostic_columns: empty column name"
            raise ValueError(msg)
        if len(set(v)) != len(v):
            msg = f"m9.diagnostic_columns must be distinct, got {v}"
            raise ValueError(msg)
        return v
