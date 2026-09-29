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

from pathlib import Path  # noqa: TC003 — Pydantic resolves field annotations at runtime
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from rux_ml.config._strict_model import StrictModel


class M9GatesConfig(StrictModel):
    """Where the operator-signed M9 keys live and who must have signed them (PR-044).

    The keys are in the rux-capital program's ``config/gates.yaml``, signed with
    ``ssh-keygen -Y sign``. The signer identity, namespace and allowed-signers
    file come from here — never from the signed file itself, which could name its
    own signer. Every field is required: no default location, no default signer.
    """

    path: Path
    signature_path: Path
    allowed_signers: Path
    identity: str = Field(min_length=1)
    namespace: str = Field(min_length=1)


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

    # PR-054 (program PR-027 A4, operator ruling 2026-09-28): the row's identity for the
    # row-random regime. A row's partition is a keyed hash of these columns (splitmix64
    # over the integer-encoded keys, seeded by the split seed — ``rux_ml.data.partitions``),
    # not a shuffle of the frame, so it is reproducible from the row alone. Required when
    # ``data.split_kind == "random"``; a null / NaN key refuses the split.
    row_key_columns: list[str] = Field(default_factory=list)

    # PR-042: h_max, in ``data.time_column`` units (epoch ms) = the label's REACH — the
    # latest price any label of a row stamped t reads, t + h_max (program PR-024 A3:
    # entry latency + the order's resting horizon + the longest markout horizon).
    # Program D41 / D45 #4: the time-block regime's embargo >= h_max. Required
    # whenever ``data.split_kind == "time_ordered"`` (fail closed).
    h_max_ms: int | None = Field(default=None, gt=0)

    # PR-044 (program D45 #3): markout and walk are scored by MAE in bp "with the
    # signed-error honesty test"; the fill fraction by Brier alone. True runs the
    # test on the held-out partition at the operator-signed thresholds, which
    # requires ``gates`` (fail closed: refused at load without it).
    signed_error_honesty: bool = False
    gates: M9GatesConfig | None = None

    # PR-046 (program D43): the learning-curve report. The nested prefixes (as
    # fractions of the train window), the OOS value it follows — a dotted path into
    # a fit's ``fold_meta.json`` ``oos`` record, e.g. "score" — and whether lower or
    # higher is better. Required by ``rux-ml runs learning-curve``, not by ``train``.
    learning_curve_fractions: list[float] | None = None
    learning_curve_metric: str | None = None
    learning_curve_direction: Literal["minimize", "maximize"] | None = None

    @model_validator(mode="after")
    def _honesty_needs_the_signed_keys(self) -> M9Config:
        if self.signed_error_honesty and self.gates is None:
            msg = "m9.signed_error_honesty needs [m9.gates] (the operator-signed thresholds)"
            raise ValueError(msg)
        return self

    @field_validator("diagnostic_columns", "row_filter_non_null", "row_key_columns")
    @classmethod
    def _distinct_non_empty(cls, v: list[str], info: ValidationInfo) -> list[str]:
        if any(not c for c in v):
            msg = f"m9.{info.field_name}: empty column name"
            raise ValueError(msg)
        if len(set(v)) != len(v):
            msg = f"m9.{info.field_name} must be distinct, got {v}"
            raise ValueError(msg)
        return v
