"""Unit tests for the Splitter Protocol + concrete strategies (per PR-015)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rux_ml.config import (
    CombinatorialPurgedCV,
    CVConfig,
    GroupKFoldCV,
    KFoldCV,
    StratifiedKFoldCV,
    TimeSeriesSplitCV,
)
from rux_ml.data import (
    CombinatorialPurgedSplitter,
    GroupKFoldSplitter,
    KFoldSplitter,
    Splitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    make_splitter,
)


@pytest.fixture
def df_small() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pl.DataFrame(
        {
            "x1": rng.normal(size=n).tolist(),
            "x2": rng.normal(size=n).tolist(),
            "y": rng.integers(0, 2, size=n).tolist(),
            "g": [(i // 10) for i in range(n)],  # 6 groups of 10
        }
    )


def _accept(_splitter: Splitter) -> None:
    """Static-typing acceptance: basedpyright fails if the concrete class
    doesn't structurally conform to the Splitter Protocol."""


# ---------- KFoldSplitter ----------


def test_kfold_yields_n_splits_pairs(df_small: pl.DataFrame) -> None:
    sp = KFoldSplitter(n_splits=5, shuffle=True, seed=0)
    _accept(sp)
    splits = list(sp.split(df_small))
    assert len(splits) == 5
    assert sp.get_n_splits() == 5
    all_test: list[int] = []
    for train, test in splits:
        # disjoint
        assert len(set(train) & set(test)) == 0
        all_test.extend(test.tolist())
    # Every row appears in exactly one test fold
    assert sorted(all_test) == list(range(df_small.height))


def test_kfold_deterministic_with_same_seed(df_small: pl.DataFrame) -> None:
    a = list(KFoldSplitter(n_splits=5, shuffle=True, seed=42).split(df_small))
    b = list(KFoldSplitter(n_splits=5, shuffle=True, seed=42).split(df_small))
    for (ta, te_a), (tb, te_b) in zip(a, b, strict=True):
        assert np.array_equal(ta, tb)
        assert np.array_equal(te_a, te_b)


def test_kfold_differs_for_different_seeds(df_small: pl.DataFrame) -> None:
    a = list(KFoldSplitter(n_splits=5, shuffle=True, seed=1).split(df_small))
    b = list(KFoldSplitter(n_splits=5, shuffle=True, seed=2).split(df_small))
    assert not all(np.array_equal(te_a, te_b) for (_, te_a), (_, te_b) in zip(a, b, strict=True))


def test_kfold_not_extmem_compatible() -> None:
    assert KFoldSplitter.extmem_compatible is False


# ---------- StratifiedKFoldSplitter ----------


def test_stratified_requires_y(df_small: pl.DataFrame) -> None:
    sp = StratifiedKFoldSplitter(n_splits=3, shuffle=True, seed=0)
    _accept(sp)
    with pytest.raises(ValueError, match="requires y"):
        list(sp.split(df_small))


def test_stratified_preserves_class_balance(df_small: pl.DataFrame) -> None:
    sp = StratifiedKFoldSplitter(n_splits=3, shuffle=True, seed=0)
    y = df_small["y"]
    overall_rate = float(y.mean())  # type: ignore[arg-type]
    for _, test_idx in sp.split(df_small, y):
        fold_rate = float(np.mean(y.to_numpy()[test_idx]))
        # Allow generous tolerance for tiny folds
        assert abs(fold_rate - overall_rate) < 0.3


def test_stratified_not_extmem_compatible() -> None:
    assert StratifiedKFoldSplitter.extmem_compatible is False


# ---------- TimeSeriesSplitter ----------


def test_time_series_train_precedes_test(df_small: pl.DataFrame) -> None:
    sp = TimeSeriesSplitter(n_splits=5, gap=0, max_train_size=None)
    _accept(sp)
    for train, test in sp.split(df_small):
        assert int(train.max()) < int(test.min())


def test_time_series_gap_honored(df_small: pl.DataFrame) -> None:
    sp = TimeSeriesSplitter(n_splits=5, gap=3, max_train_size=None)
    for train, test in sp.split(df_small):
        # gap=3 means at least 3 indices between max(train) and min(test)
        assert int(test.min()) - int(train.max()) > 3


def test_time_series_rolling_caps_train_size(df_small: pl.DataFrame) -> None:
    sp = TimeSeriesSplitter(n_splits=3, gap=0, max_train_size=10)
    for train, _ in sp.split(df_small):
        assert len(train) <= 10


def test_time_series_is_extmem_compatible() -> None:
    """Only Splitter with extmem_compatible=True in v0."""
    assert TimeSeriesSplitter.extmem_compatible is True


# ---------- GroupKFoldSplitter ----------


def test_group_kfold_requires_groups(df_small: pl.DataFrame) -> None:
    sp = GroupKFoldSplitter(n_splits=3)
    _accept(sp)
    with pytest.raises(ValueError, match="requires `groups`"):
        list(sp.split(df_small))


def test_group_kfold_no_group_overlap(df_small: pl.DataFrame) -> None:
    sp = GroupKFoldSplitter(n_splits=3)
    groups = df_small["g"].to_numpy()
    for train, test in sp.split(df_small, groups=groups):
        train_groups = set(groups[train].tolist())
        test_groups = set(groups[test].tolist())
        assert not (train_groups & test_groups)


def test_group_kfold_not_extmem_compatible() -> None:
    assert GroupKFoldSplitter.extmem_compatible is False


# ---------- CombinatorialPurgedSplitter ----------


def test_cpcv_yields_combinations(df_small: pl.DataFrame) -> None:
    """C(5, 2) = 10 yielded splits."""
    sp = CombinatorialPurgedSplitter(n_folds=5, n_test_folds=2, purged_size=0, embargo_size=0)
    _accept(sp)
    splits = list(sp.split(df_small))
    assert len(splits) == 10
    assert sp.get_n_splits() == 10
    for train, test in splits:
        assert isinstance(train, np.ndarray)
        assert isinstance(test, np.ndarray)
        assert len(set(train) & set(test)) == 0


def test_cpcv_not_extmem_compatible() -> None:
    assert CombinatorialPurgedSplitter.extmem_compatible is False


# ---------- make_splitter factory ----------


@pytest.mark.parametrize(
    ("cfg", "expected_cls"),
    [
        (KFoldCV(n_splits=3), KFoldSplitter),
        (StratifiedKFoldCV(n_splits=3), StratifiedKFoldSplitter),
        (TimeSeriesSplitCV(n_splits=3), TimeSeriesSplitter),
        (GroupKFoldCV(n_splits=3, groups_column="g"), GroupKFoldSplitter),
        (
            CombinatorialPurgedCV(n_folds=5, n_test_folds=2, purged_size=0, embargo_size=0),
            CombinatorialPurgedSplitter,
        ),
    ],
)
def test_make_splitter_dispatches_by_kind(cfg: CVConfig, expected_cls: type) -> None:
    sp = make_splitter(cfg, seed=7)
    assert isinstance(sp, expected_cls)


def test_make_splitter_seed_threads_into_kfold(df_small: pl.DataFrame) -> None:
    a = list(make_splitter(KFoldCV(n_splits=3, shuffle=True), seed=11).split(df_small))
    b = list(make_splitter(KFoldCV(n_splits=3, shuffle=True), seed=11).split(df_small))
    for (ta, _), (tb, _) in zip(a, b, strict=True):
        assert np.array_equal(ta, tb)
