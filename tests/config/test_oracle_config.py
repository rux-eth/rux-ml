"""``[data.oracle]`` config (PR-040): loading, overrides, validators, hash neutrality.

The pinned hashes below were computed at dev 679b096 (before PR-040) with the
unmodified config code. ``oracle`` is hash-elided (D17 widened to output-neutral
guard config), so adding the field must not move ``data_cfg_hash`` or
``root_cfg_hash`` for any existing config — set or unset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rux_ml.config.data import DataConfig, OracleQuarantineConfig
from rux_ml.config.root import RuxMLConfig, cfg_hash, layer_cfg_hash

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# dev 679b096, configs/base.toml -> problems/crypto_breakout_h3 -> studies/crypto_breakout_h3_hpo
HPO_DATA_CFG_HASH = "3c89c1f5ea24b975e78a101c0a64c73d1c27aa78609965adb6284238d7c96435"
HPO_ROOT_CFG_HASH = "06f13e9ca250208df80978855484d215e78783fda33472acede2fddf5e649331"
# dev 679b096, RuxMLConfig(data=DataConfig(source_path="x.parquet", target_column="y"))
MIN_DATA_CFG_HASH = "2d7467a5cd1bd0e3c917735ed06362ef7875a9ba8ac7afd85474142f9dda4f77"
MIN_ROOT_CFG_HASH = "83c3090f6da786552d4322dec8f638eb5fdc554b2ad70c23a3a0809955945bc8"


def _hpo(**kw: object) -> RuxMLConfig:
    return RuxMLConfig.from_layers(
        REPO_CONFIGS / "base.toml",
        problem="crypto_breakout_h3",
        study="crypto_breakout_h3_hpo",
        problems_dir=REPO_CONFIGS / "problems",
        studies_dir=REPO_CONFIGS / "studies",
        **kw,  # type: ignore[arg-type]
    )


def test_repo_base_toml_carries_the_oracle_table(oracle_values: dict[str, str]) -> None:
    oracle = _hpo().data.oracle
    assert oracle is not None
    assert oracle.model_dump() == oracle_values


def test_repo_config_hashes_unchanged_by_the_oracle_table() -> None:
    cfg = _hpo()
    assert layer_cfg_hash(cfg, "data") == HPO_DATA_CFG_HASH
    assert cfg_hash(cfg) == HPO_ROOT_CFG_HASH


def test_repo_config_hashes_unchanged_when_oracle_values_are_overridden() -> None:
    cfg = _hpo(overrides={"data.oracle.namespace": "other__"})
    assert cfg.data.oracle is not None
    assert cfg.data.oracle.namespace == "other__"
    assert layer_cfg_hash(cfg, "data") == HPO_DATA_CFG_HASH
    assert cfg_hash(cfg) == HPO_ROOT_CFG_HASH


@pytest.mark.parametrize("with_oracle", [False, True])
def test_direct_config_hashes_unchanged_set_or_unset(
    oracle_cfg: OracleQuarantineConfig, *, with_oracle: bool
) -> None:
    cfg = RuxMLConfig(
        data=DataConfig(
            source_path=Path("x.parquet"),
            target_column="y",
            oracle=oracle_cfg if with_oracle else None,
        )
    )
    assert layer_cfg_hash(cfg, "data") == MIN_DATA_CFG_HASH
    assert cfg_hash(cfg) == MIN_ROOT_CFG_HASH


def test_env_override_reaches_the_nested_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUXML_DATA__ORACLE__NAMESPACE", "envns__")
    cfg = _hpo()
    assert cfg.data.oracle is not None
    assert cfg.data.oracle.namespace == "envns__"


def test_dump_validate_rebuild_keeps_the_values(oracle_values: dict[str, str]) -> None:
    """The rebuild shape ``tuning/objective.py`` and ``registry/promote.py`` use."""
    cfg = _hpo()
    rebuilt = RuxMLConfig.model_validate(cfg.model_dump())
    assert rebuilt.data.oracle is not None
    assert rebuilt.data.oracle.model_dump() == oracle_values


@pytest.mark.parametrize(
    "bad",
    [
        {"namespace": "", "tag_file": "T.tag"},
        {"namespace": "o__", "tag_file": ""},
        {"namespace": "o__", "tag_file": "a/T.tag"},
        {"namespace": "o__", "tag_file": "a\\T.tag"},
        {"namespace": "o__"},
    ],
)
def test_validators_reject_values_that_would_disable_the_check(bad: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        OracleQuarantineConfig(**bad)
