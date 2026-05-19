"""Tests for ``CVConfig`` tagged union + hash integration with ``RuxMLConfig``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rux_ml.config import (
    CombinatorialPurgedCV,
    DataConfig,
    GroupKFoldCV,
    KFoldCV,
    RuxMLConfig,
    StratifiedKFoldCV,
    TimeSeriesSplitCV,
    cfg_hash,
    layer_cfg_hash,
)


def test_default_cv_is_kfold() -> None:
    cfg = RuxMLConfig()
    assert isinstance(cfg.cv, KFoldCV)
    assert cfg.cv.kind == "kfold"
    assert cfg.cv.n_splits == 5


# PR-024: temporal CV kinds (time_series, cpcv, panel_cpcv) require the
# baseline path to use time_ordered splits too — the validator rejects
# otherwise. Each row carries the [data] block needed to satisfy that rule
# (None for non-temporal kinds keeps the test surface small).
_TEMPORAL_DATA_BLOCK = '[data]\nsplit_kind = "time_ordered"\ntime_column = "ts"\n'


@pytest.mark.parametrize(
    ("kind", "expected_cls", "extra_toml", "data_block"),
    [
        ("kfold", KFoldCV, "", ""),
        ("stratified_kfold", StratifiedKFoldCV, "", ""),
        ("time_series", TimeSeriesSplitCV, "gap = 2\n", _TEMPORAL_DATA_BLOCK),
        ("group_kfold", GroupKFoldCV, 'groups_column = "g"\n', ""),
        ("cpcv", CombinatorialPurgedCV, "n_folds = 6\nn_test_folds = 2\n", _TEMPORAL_DATA_BLOCK),
    ],
)
def test_cv_config_loads_each_kind_from_toml(
    tmp_path: Path,
    kind: str,
    expected_cls: type,
    extra_toml: str,
    data_block: str,
) -> None:
    config = tmp_path / "base.toml"
    config.write_text(f'{data_block}[cv]\nkind = "{kind}"\n{extra_toml}')
    cfg = RuxMLConfig.from_layers(config)
    assert isinstance(cfg.cv, expected_cls)
    assert cfg.cv.kind == kind


def test_cv_config_rejects_unknown_kind(tmp_path: Path) -> None:
    config = tmp_path / "base.toml"
    config.write_text('[cv]\nkind = "bogus"\n')
    with pytest.raises(Exception, match=r"discriminator|bogus|kind"):
        RuxMLConfig.from_layers(config)


def test_cv_config_rejects_unknown_field_per_extra_forbid() -> None:
    with pytest.raises(Exception, match="extra"):
        KFoldCV(n_splits=5, shuffle=True, unknown_field=1)  # type: ignore[call-arg]


def test_group_kfold_requires_groups_column() -> None:
    """groups_column has no default — group splitting without it is meaningless."""
    with pytest.raises(Exception, match="groups_column"):
        GroupKFoldCV(n_splits=5)  # type: ignore[call-arg]


def test_layer_cfg_hash_for_cv_layer_changes_with_kind() -> None:
    cfg_a = RuxMLConfig(cv=KFoldCV(n_splits=5))
    cfg_b = RuxMLConfig(cv=KFoldCV(n_splits=10))
    cfg_c = RuxMLConfig(cv=StratifiedKFoldCV(n_splits=5))
    h_a = layer_cfg_hash(cfg_a, "cv")
    h_b = layer_cfg_hash(cfg_b, "cv")
    h_c = layer_cfg_hash(cfg_c, "cv")
    assert h_a != h_b  # n_splits differs
    assert h_a != h_c  # kind differs
    assert len(h_a) == 64  # sha256 hex


def test_root_cfg_hash_changes_when_cv_changes() -> None:
    h_a = cfg_hash(RuxMLConfig(cv=KFoldCV(n_splits=5)))
    h_b = cfg_hash(
        RuxMLConfig(
            data=DataConfig(split_kind="time_ordered", time_column="ts"),
            cv=TimeSeriesSplitCV(n_splits=5, gap=0, max_train_size=None),
        )
    )
    assert h_a != h_b


def test_layer_cfg_hash_rejects_unknown_layer_name() -> None:
    cfg = RuxMLConfig()
    with pytest.raises(ValueError, match="unknown layer"):
        layer_cfg_hash(cfg, "bogus")
