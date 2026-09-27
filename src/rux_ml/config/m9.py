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

from pydantic import Field, ValidationInfo, field_validator

from rux_ml.config._strict_model import StrictModel


class M9Config(StrictModel):
    """Per-problem M9 settings. Later PRs of the R-14 track add ``honesty`` and
    ``learning_curve`` sub-tables here."""

    # Labels carried beside the target as diagnostics (distinct, non-empty names).
    diagnostic_columns: list[str] = Field(default_factory=list)

    # The row predicate applied before the split (program PR-024 A9): a row with a
    # null (or NaN) in any listed column is dropped, so it is neither trained nor
    # scored. The markout problem lists its own target — ``y__markout_bp`` is null
    # on an unfilled order — and trains and scores on filled rows only. Empty keeps
    # every row. A listed column absent from the set refuses the split.
    row_filter_non_null: list[str] = Field(default_factory=list)

    # PR-042: the longest label horizon, in ``data.time_column`` units (epoch ms).
    # Program D41 / D45 #4: the time-block regime's embargo >= h_max. Required
    # whenever ``data.split_kind == "time_ordered"`` (fail closed).
    h_max_ms: int | None = Field(default=None, gt=0)

    @field_validator("diagnostic_columns", "row_filter_non_null")
    @classmethod
    def _distinct_non_empty(cls, v: list[str], info: ValidationInfo) -> list[str]:
        if any(not c for c in v):
            msg = f"m9.{info.field_name}: empty column name"
            raise ValueError(msg)
        if len(set(v)) != len(v):
            msg = f"m9.{info.field_name} must be distinct, got {v}"
            raise ValueError(msg)
        return v
