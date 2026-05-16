"""CV-strategy config (per PR-015 — Tier-2 research-backed, Synthesis 2026-05-16).

Tagged-union over the five concrete Splitter strategies the workbench supports
at v0:

- ``KFoldCV`` — IID baseline (sklearn ``KFold``)
- ``StratifiedKFoldCV`` — class-balance-preserving (sklearn ``StratifiedKFold``)
- ``TimeSeriesSplitCV`` — walk-forward (sklearn ``TimeSeriesSplit`` with ``gap``
  and optional ``max_train_size`` for the rolling-vs-expanding rule)
- ``GroupKFoldCV`` — group-leakage-preventing (sklearn ``GroupKFold``);
  ``groups_column`` references a column in the input DataFrame and is resolved
  to ``np.ndarray`` at the data-layer boundary (column-to-array pattern)
- ``CombinatorialPurgedCV`` — AFML CPCV with one-sided post-test embargo
  + two-sided label-overlap purge (wraps ``skfolio.model_selection.CombinatorialPurgedCV``)

Per ``docs/CONSTRAINTS.md`` "Zero Hardcoded Parameters", the embargo / purge
defaults are 0 (= "no embargo", a meaningful operational state matching both
``skfolio`` and ``timeseriescv``); users opt into AFML-recommended values
``embargo_size ∈ [0.005, 0.02] · N`` per problem.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from rux_ml.config._strict_model import StrictModel


class KFoldCV(StrictModel):
    """IID K-fold cross-validation (sklearn ``KFold``)."""

    kind: Literal["kfold"] = "kfold"
    n_splits: int = 5
    shuffle: bool = True


class StratifiedKFoldCV(StrictModel):
    """Class-balance-preserving K-fold (sklearn ``StratifiedKFold``)."""

    kind: Literal["stratified_kfold"] = "stratified_kfold"
    n_splits: int = 5
    shuffle: bool = True


class TimeSeriesSplitCV(StrictModel):
    """Walk-forward (sklearn ``TimeSeriesSplit``).

    ``max_train_size`` selects between expanding (``None`` — default) and
    rolling (``int``) windows per the Q1.b research finding. ``gap`` excludes
    samples between train-end and test-start for simple label-horizon leakage
    (per Q2.a — sufficient when label horizon is fixed and ≤ ``gap`` bars).
    """

    kind: Literal["time_series"] = "time_series"
    n_splits: int = 5
    gap: int = 0
    max_train_size: int | None = None


class GroupKFoldCV(StrictModel):
    """Group-leakage-preventing K-fold (sklearn ``GroupKFold``).

    ``groups_column`` names a column in the input DataFrame; the data layer
    resolves it to ``np.ndarray`` at the Splitter call site (column-to-array
    convention per Q1.c, two cited production precedents: sklearn user guide,
    mlxtend GroupTimeSeriesSplit user guide).
    """

    kind: Literal["group_kfold"] = "group_kfold"
    n_splits: int = 5
    groups_column: str  # No default: group splitting without a groups column is meaningless.


class CombinatorialPurgedCV(StrictModel):
    """AFML CPCV with purge + one-sided post-test embargo.

    Wraps ``skfolio.model_selection.CombinatorialPurgedCV`` (BSD-3, v0.20.1+).
    AFML-recommended embargo range is ``h ∈ [0.005, 0.02]·N``; users set
    ``embargo_size`` (integer count of bars) per problem. Default ``0`` means
    "no embargo" — a valid operational state when labels are point-in-time.
    """

    kind: Literal["cpcv"] = "cpcv"
    n_folds: int = 10
    n_test_folds: int = 2
    purged_size: int = 0
    embargo_size: int = 0


CVConfig = Annotated[
    KFoldCV | StratifiedKFoldCV | TimeSeriesSplitCV | GroupKFoldCV | CombinatorialPurgedCV,
    Field(discriminator="kind"),
]
