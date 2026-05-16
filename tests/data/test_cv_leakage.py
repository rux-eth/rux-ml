"""Leakage-prevention tests per PR-015 Phase 3 findings (Q2.a, Q2.b, Q2.c)."""

from __future__ import annotations

import numpy as np
import polars as pl

from rux_ml.data import (
    CombinatorialPurgedSplitter,
    GroupKFoldSplitter,
    TimeSeriesSplitter,
)


def _df(n: int) -> pl.DataFrame:
    return pl.DataFrame({"x": list(range(n))})


# ---------- Time-leakage (Q2.a) ----------


def test_time_series_no_time_leakage_with_gap() -> None:
    """TimeSeriesSplit + gap=k: max(train) + k < min(test) for every fold."""
    gap = 5
    sp = TimeSeriesSplitter(n_splits=4, gap=gap, max_train_size=None)
    for train, test in sp.split(_df(120)):
        assert int(test.min()) - int(train.max()) > gap


def test_time_series_indices_monotonic() -> None:
    """Train indices always precede test indices in every fold."""
    sp = TimeSeriesSplitter(n_splits=5, gap=0, max_train_size=None)
    for train, test in sp.split(_df(100)):
        assert int(train.max()) < int(test.min())


# ---------- Group-leakage (Q2.b) ----------


def test_group_kfold_no_group_id_spans_train_and_test() -> None:
    """Per Q2.b: GroupKFold guarantees each group appears in exactly one test fold."""
    n = 100
    # 10 groups of 10 rows each.
    groups = np.repeat(np.arange(10), 10)
    sp = GroupKFoldSplitter(n_splits=5)
    seen_test_groups: set[int] = set()
    for train, test in sp.split(_df(n), groups=groups):
        train_groups = set(groups[train].tolist())
        test_groups = set(groups[test].tolist())
        assert not (train_groups & test_groups)
        # No group appears in more than one test fold
        assert not (seen_test_groups & test_groups)
        seen_test_groups.update(test_groups)
    # Every group ended up in exactly one test fold
    assert seen_test_groups == set(range(10))


# ---------- CPCV embargo + purge (Q2.c) ----------


def test_cpcv_embargo_purges_post_test_observations() -> None:
    """Per Q2.c (AFML §7.4.2): embargo_size purges training observations
    within ``embargo_size`` indices AFTER each test fold boundary."""
    n = 200
    embargo = 5
    sp = CombinatorialPurgedSplitter(
        n_folds=10,
        n_test_folds=2,
        purged_size=0,
        embargo_size=embargo,
    )
    for train, test in sp.split(_df(n)):
        if test.size == 0:
            continue
        # Group test indices into contiguous test-fold spans.
        test_sorted = np.sort(test)
        breaks = np.where(np.diff(test_sorted) > 1)[0]
        # Per-span boundaries: each span's max index is a "test fold end"
        spans = np.split(test_sorted, breaks + 1)
        for span in spans:
            test_end = int(span.max())
            # No training observation may fall in (test_end, test_end + embargo]
            forbidden_zone = set(range(test_end + 1, test_end + embargo + 1))
            train_set = set(train.tolist())
            overlap = forbidden_zone & train_set
            assert not overlap, (
                f"embargo violation: train indices {sorted(overlap)} "
                f"in forbidden zone (test_end={test_end}, embargo={embargo})"
            )


def test_cpcv_purge_removes_overlapping_pre_test_observations() -> None:
    """Per Q2.c: purged_size removes training observations BEFORE each test fold
    (label-overlap purge — symmetric in implementations, here we verify the
    pre-test side which skfolio handles when purged_size > 0)."""
    n = 200
    purged = 5
    sp = CombinatorialPurgedSplitter(
        n_folds=10,
        n_test_folds=2,
        purged_size=purged,
        embargo_size=0,
    )
    for train, test in sp.split(_df(n)):
        if test.size == 0:
            continue
        test_sorted = np.sort(test)
        breaks = np.where(np.diff(test_sorted) > 1)[0]
        spans = np.split(test_sorted, breaks + 1)
        for span in spans:
            test_start = int(span.min())
            # No training observation in [test_start - purged, test_start)
            forbidden_zone = set(range(max(0, test_start - purged), test_start))
            train_set = set(train.tolist())
            overlap = forbidden_zone & train_set
            assert not overlap, (
                f"purge violation: train indices {sorted(overlap)} "
                f"in pre-test zone (test_start={test_start}, purged={purged})"
            )
