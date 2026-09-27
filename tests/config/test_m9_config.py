"""The optional ``[m9]`` layer (PR-041): validators and hash behaviour.

Pinned hashes are the PR-040 pins (dev 679b096). An optional layer that is unset
contributes nothing to ``root_cfg_hash``, so every existing config keeps its
recorded hashes; once set, the layer is part of the trial's identity and moves
the root hash.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rux_ml.config import DataConfig, FeaturesConfig, FeaturesSpec, M9Config, RuxMLConfig
from rux_ml.config.root import cfg_hash, layer_cfg_hash

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
HPO_ROOT_CFG_HASH = "06f13e9ca250208df80978855484d215e78783fda33472acede2fddf5e649331"
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


def test_unset_layer_leaves_every_pinned_root_hash_unchanged() -> None:
    assert _hpo().m9 is None
    assert cfg_hash(_hpo()) == HPO_ROOT_CFG_HASH
    minimal = RuxMLConfig(data=DataConfig(source_path=Path("x.parquet"), target_column="y"))
    assert minimal.m9 is None
    assert cfg_hash(minimal) == MIN_ROOT_CFG_HASH


def test_set_layer_is_part_of_the_trial_identity() -> None:
    # The hpo study splits time_ordered, so [m9] also needs h_max_ms and a covering
    # embargo (PR-042's fail-closed rule).
    cfg = _hpo(
        overrides={
            "m9.diagnostic_columns": ["y__other"],
            "m9.h_max_ms": 1,
            "data.split_embargo": 1,
        }
    )
    assert cfg.m9 is not None
    assert cfg.m9.diagnostic_columns == ["y__other"]
    assert cfg_hash(cfg) != HPO_ROOT_CFG_HASH
    assert layer_cfg_hash(cfg, "m9") != layer_cfg_hash(_hpo(), "m9")


def test_layer_loads_from_toml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "base.toml"
    p.write_text('[data]\ntarget_column = "y"\n[m9]\ndiagnostic_columns = ["a", "b"]\n')
    assert RuxMLConfig.from_layers(p).m9 == M9Config(diagnostic_columns=["a", "b"])
    monkeypatch.setenv("RUXML_M9__DIAGNOSTIC_COLUMNS", '["c"]')
    assert RuxMLConfig.from_layers(p).m9 == M9Config(diagnostic_columns=["c"])


def test_target_may_not_be_a_diagnostic() -> None:
    with pytest.raises(ValidationError, match="target_column"):
        RuxMLConfig(data=DataConfig(target_column="y"), m9=M9Config(diagnostic_columns=["y", "z"]))


def test_diagnostics_among_the_features_are_refused_at_build_not_at_load() -> None:
    """PR-043 moved this rule into ``check_feature_labels`` so the CLI exits 2 on it."""
    from rux_ml.data import LabelAsFeatureError, check_feature_labels  # noqa: PLC0415
    from tests.conftest import repo_oracle_cfg  # noqa: PLC0415

    cfg = RuxMLConfig(
        data=DataConfig(target_column="y", oracle=repo_oracle_cfg()),
        features=FeaturesConfig(spec=FeaturesSpec(numeric_columns=["x", "z"])),
        m9=M9Config(diagnostic_columns=["z"]),
    )
    with pytest.raises(LabelAsFeatureError, match="'z'"):
        check_feature_labels(cfg)


def test_diagnostics_must_be_distinct_non_empty_names() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        M9Config(diagnostic_columns=["z", "z"])
    with pytest.raises(ValidationError):
        M9Config(diagnostic_columns=[""])


# ---------- m9.row_filter_non_null (program PR-024 A9) ----------


def test_row_filter_columns_must_be_distinct_and_named() -> None:
    with pytest.raises(ValidationError, match="row_filter_non_null"):
        M9Config(row_filter_non_null=["y", "y"])
    with pytest.raises(ValidationError, match="row_filter_non_null"):
        M9Config(row_filter_non_null=[""])
    assert M9Config(row_filter_non_null=["y"]).row_filter_non_null == ["y"]
