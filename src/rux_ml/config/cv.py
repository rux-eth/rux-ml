"""CV-strategy config (per PR-015 - Tier-2 research-backed, Synthesis 2026-05-16;
PR-023 - D1-D5 panel-aware extensions, 2026-05-19).

Tagged-union over the six concrete Splitter strategies the workbench supports:

- ``KFoldCV`` — IID baseline (sklearn ``KFold``)
- ``StratifiedKFoldCV`` — class-balance-preserving (sklearn ``StratifiedKFold``)
- ``TimeSeriesSplitCV`` — walk-forward (sklearn ``TimeSeriesSplit`` with ``gap``
  and optional ``max_train_size`` for the rolling-vs-expanding rule). PR-023
  extends with polymorphic ``embargo_time: int | str`` + ``time_column`` for
  time-unit embargo on stacked panels (D2).
- ``GroupKFoldCV`` — group-leakage-preventing (sklearn ``GroupKFold``);
  ``groups_column`` references a column in the input DataFrame and is resolved
  to ``np.ndarray`` at the data-layer boundary (column-to-array pattern)
- ``CombinatorialPurgedCV`` — AFML CPCV with one-sided post-test embargo
  + two-sided label-overlap purge (wraps ``skfolio.model_selection.CombinatorialPurgedCV``).
  PR-023 surfaces ergonomic ``target_horizon_bars`` + ``embargo_pct`` knobs that
  convert internally to skfolio's row-count args (D3, AFML Snippet 7.3).
- ``PanelCombinatorialPurgedCV`` — panel-aware CPCV (PR-023 D1). Folds over
  UNIQUE sorted timestamps from ``time_column``; per-asset (``asset_column``)
  purge in timestamp units; combinatorial paths preserved via post-hoc row-
  index decomposition. Convention: mlfinlab ``StackedCombinatorialPurgedKFold``
  + Numerai era-CV.

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

    **Panel-aware embargo (PR-023 D2)**: ``embargo_time`` is polymorphic — set
    to an ``int`` for row-count semantics (matches ``gap``; default ``None``
    falls back to ``gap``) or to a duration string like ``"24h"`` parsed via
    ``pandas.Timedelta`` for time-unit semantics. The time-unit path requires
    ``time_column`` to be set; the splitter then translates the requested
    duration into row counts by inspecting the input DataFrame's timestamps.

    **Int64 timestamp handling (PR-027)**: ``time_unit`` declares how to
    interpret ``time_column`` when the column dtype is ``pl.Int64`` (e.g.,
    Unix-seconds from a database column). When the column is already
    ``pl.Datetime``, the column carries its own unit; setting ``time_unit``
    on a ``pl.Datetime`` column raises ``ValueError``. When the column is
    ``pl.Int64``, ``time_unit`` MUST be set or the splitter raises
    ``ValueError`` at split time. Convention inherited from polars
    ``pl.from_epoch(time_unit=...)`` and pandas ``pd.to_datetime(unit=...)``
    (4-of-6 surveyed CV / time-series libraries enforce explicit unit
    declaration; see PR-027 Phase 3).

    Defaults preserve v0.1 row-count behavior. Single-asset time-aligned
    panels should leave both fields ``None``; stacked panels with many rows
    per timestamp set ``time_column`` + ``embargo_time`` (+ ``time_unit`` if
    the column is ``pl.Int64``).
    """

    kind: Literal["time_series"] = "time_series"
    n_splits: int = 5
    gap: int = 0
    max_train_size: int | None = None
    time_column: str | None = None
    embargo_time: int | str | None = None
    time_unit: Literal["ns", "us", "ms", "s"] | None = None


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

    **Ergonomic knobs (PR-023 D3)**: ``target_horizon_bars`` and
    ``embargo_pct`` are the AFML-Snippet-7.3 verbatim formula surfaced at the
    workbench level. When non-zero they convert internally:

    - ``purged_size = target_horizon_bars`` (symmetric, conservative-correct
      on regular bars — accepts skfolio's coarser model per D4).
    - ``embargo_size = int(N * embargo_pct)`` (AFML ``mbrg = int(X.shape[0] *
      pctEmbargo)``).

    When the ergonomic knob and its row-count sibling are both supplied,
    the row-count value wins (passes through unchanged) — the ergonomic
    knobs are a strict opt-in. ``docs/CONVENTIONS.md`` documents the
    precision gap vs AFML interval-overlap purge.
    """

    kind: Literal["cpcv"] = "cpcv"
    n_folds: int = 10
    n_test_folds: int = 2
    purged_size: int = 0
    embargo_size: int = 0
    target_horizon_bars: int = 0
    embargo_pct: float = 0.0


class PanelCombinatorialPurgedCV(StrictModel):
    """Panel-aware CPCV (PR-023 D1).

    Folds are computed over the **unique sorted timestamps** read from
    ``time_column`` — each fold over the time axis maps to all asset rows at
    those timestamps. ``target_horizon_bars`` and ``embargo_pct`` are
    interpreted in **timestamp units** (the fold axis) rather than row units;
    that's the point of D1 (``gap`` collapsing to <1 timestamp on stacked
    panels was the motivating bug). Per-asset purge is enforced by the
    timestamp-fold mapping: dropping a timestamp from train removes every
    asset row at that timestamp.

    ``asset_column`` is required (and validated at split time) so the
    splitter can decompose combinatorial paths via row-index ``groupby``.
    Convention: mlfinlab ``StackedCombinatorialPurgedKFold`` + Numerai era-CV.

    Defaults match ``CombinatorialPurgedCV`` (no embargo / no purge); users
    opt into AFML-recommended values per problem.
    """

    kind: Literal["panel_cpcv"] = "panel_cpcv"
    n_folds: int = 10
    n_test_folds: int = 2
    time_column: str
    asset_column: str
    target_horizon_bars: int = 0
    embargo_pct: float = 0.0


CVConfig = Annotated[
    KFoldCV
    | StratifiedKFoldCV
    | TimeSeriesSplitCV
    | GroupKFoldCV
    | CombinatorialPurgedCV
    | PanelCombinatorialPurgedCV,
    Field(discriminator="kind"),
]
