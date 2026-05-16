"""Tests for cfg_hash / layer_cfg_hash semantics (per D17)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rux_ml.config import RuxMLConfig, cfg_hash, layer_cfg_hash


def test_cfg_hash_is_deterministic_across_identical_loads(base_toml: Path) -> None:
    a = cfg_hash(RuxMLConfig.from_layers(base_toml))
    b = cfg_hash(RuxMLConfig.from_layers(base_toml))
    assert a == b


def test_cfg_hash_differs_when_a_non_elided_field_changes(
    base_toml: Path,
) -> None:
    a = cfg_hash(RuxMLConfig.from_layers(base_toml))
    b = cfg_hash(RuxMLConfig.from_layers(base_toml, overrides={"training.learning_rate": 0.42}))
    assert a != b


def test_cfg_hash_is_stable_when_only_elided_fields_change(
    base_toml: Path,
) -> None:
    a = cfg_hash(RuxMLConfig.from_layers(base_toml))
    # cas_root and runs.storage_url are in _HASH_ELIDED_FIELDS
    b = cfg_hash(
        RuxMLConfig.from_layers(
            base_toml,
            overrides={
                "data.cas_root": "/tmp/elsewhere/cas",
                "runs.storage_url": "sqlite:///elsewhere.db",
            },
        )
    )
    assert a == b


def test_layer_cfg_hash_only_changes_for_its_own_layer(base_toml: Path) -> None:
    base_cfg = RuxMLConfig.from_layers(base_toml)
    bumped = RuxMLConfig.from_layers(base_toml, overrides={"training.learning_rate": 0.42})

    # training layer changed → its layer hash changes
    assert layer_cfg_hash(base_cfg, "training") != layer_cfg_hash(bumped, "training")
    # other layers unchanged → their layer hashes are identical
    for other in ("data", "features", "tuning", "runs", "registry", "memory"):
        assert layer_cfg_hash(base_cfg, other) == layer_cfg_hash(bumped, other)


def test_layer_cfg_hash_rejects_unknown_layer(base_toml: Path) -> None:
    cfg = RuxMLConfig.from_layers(base_toml)
    with pytest.raises(ValueError, match="unknown layer"):
        layer_cfg_hash(cfg, "nope")


def test_layer_cfg_hash_elides_within_layer(base_toml: Path) -> None:
    base_cfg = RuxMLConfig.from_layers(base_toml)
    bumped = RuxMLConfig.from_layers(base_toml, overrides={"data.cas_root": "/tmp/somewhere"})
    # data layer's cas_root is elided -> hash unchanged
    assert layer_cfg_hash(base_cfg, "data") == layer_cfg_hash(bumped, "data")


def test_hashes_are_64_char_hex(base_toml: Path) -> None:
    cfg = RuxMLConfig.from_layers(base_toml)
    h = cfg_hash(cfg)
    assert len(h) == 64
    int(h, 16)  # parses as hex; raises if not
