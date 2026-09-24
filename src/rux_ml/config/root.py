"""Root config: composes per-layer models, loads layered TOML + env + CLI overrides.

Per D2 / D17:
- pydantic-settings v2 with `TomlConfigSettingsSource(deep_merge=True)`
- Override precedence (highest -> lowest):
  1. CLI / init_settings (passed to from_layers as kwargs)
  2. env vars (RUXML_*, with `__` for nesting)
  3. .env file
  4. study TOML
  5. problem TOML
  6. base.toml
  7. Pydantic field defaults
- Per-layer + root config hashes via `cfg_hash` / `layer_cfg_hash` (per D17),
  with elision documented in `_HASH_ELIDED_FIELDS`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from rux_ml._internal.hashing import canonical_json
from rux_ml.config.cv import CVConfig, KFoldCV
from rux_ml.config.data import DataConfig
from rux_ml.config.features import FeaturesConfig
from rux_ml.config.memory import MemoryConfig
from rux_ml.config.registry import RegistryConfig
from rux_ml.config.runs import RunsConfig
from rux_ml.config.solving import (
    SolvingConfig,  # noqa: TC001 — Pydantic needs runtime resolution for the discriminated-union annotation
)
from rux_ml.config.training import TrainingConfig, XGBoostTraining
from rux_ml.config.tuning import SearchSpec, TuningConfig

if TYPE_CHECKING:
    from collections.abc import Mapping

# CV kinds that require time-ordered one-off splits at the baseline path.
# Used by the PR-024 cross-field validator on RuxMLConfig. Tracked here so the
# rule lives next to the validator that consumes it.
_TEMPORAL_CV_KINDS: frozenset[str] = frozenset({"time_series", "cpcv", "panel_cpcv"})

# Fields elided before hashing because they are non-deterministic / runtime-only,
# or output-neutral guard config (PR-040: ``data.oracle`` only decides whether
# ingest refuses, never what a passing run computes).
# Kept here as the single source of truth so changes surface in code review.
# Per D17.
_HASH_ELIDED_FIELDS: dict[str, set[str]] = {
    "data": {"cas_root", "manifests_root", "source_path", "oracle"},
    "features": set(),
    "training": set(),
    "tuning": set(),
    "runs": {"storage_url", "artifacts_root"},
    "registry": {"root"},
    "memory": set(),
    "cv": set(),
    "solving": set(),
}


def _elide(d: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in excluded}


def _unflatten_dot_paths(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert {'training.learning_rate': 0.01} -> {'training': {'learning_rate': 0.01}}.

    Used by from_layers to translate dot-path init kwargs into nested dicts that
    the BaseSettings init_settings source then merges over the TOML stack.
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        cursor: dict[str, Any] = nested
        for part in parts[:-1]:
            sub: Any = cursor.setdefault(part, {})
            if not isinstance(sub, dict):
                msg = f"override path conflict at {part!r} in {key!r}"
                raise ValueError(msg)
            cursor = cast("dict[str, Any]", sub)
        cursor[parts[-1]] = value
    return nested


class RuxMLConfig(BaseSettings):
    """Composed root config — see module docstring for precedence."""

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=XGBoostTraining)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    runs: RunsConfig = Field(default_factory=RunsConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cv: CVConfig = Field(default_factory=KFoldCV)
    # ``solving`` is Optional (default None) — only solver-shaped runs set
    # it; Trainer-shaped runs leave it None. Per PR-020 Q-Shape (partial
    # mirror), the solving layer reuses the config / registry / Optuna
    # substrate but bypasses ``cfg.data`` and ``cfg.cv``.
    solving: SolvingConfig | None = None
    search_space: dict[str, SearchSpec] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_prefix="RUXML_",
        env_nested_delimiter="__",
        extra="forbid",
        nested_model_default_partial_update=True,
    )

    @model_validator(mode="after")
    def _validate_split_kind_consistency(self) -> RuxMLConfig:
        """PR-024 cross-field check: ``data.split_kind`` must be consistent
        with ``data.time_column`` and ``cv.kind``.

        Two rules, both fail-fast at config-load time:

        1. ``data.split_kind == "time_ordered"`` requires ``data.time_column``
           to be set — the temporal split function needs a timestamp column.
        2. ``cv.kind in {time_series, cpcv, panel_cpcv}`` requires
           ``data.split_kind == "time_ordered"`` — running a temporal CV with
           a random-shuffle one-off baseline produces meaningfully different
           train/val/test shapes between ``rux-ml train`` and ``rux-ml tune``,
           and the leakage profile of the baseline is the well-documented
           foot-gun this PR fixes.
        """
        if self.data.split_kind == "time_ordered" and self.data.time_column is None:
            msg = (
                "data.split_kind == 'time_ordered' requires data.time_column to be set "
                "(name a timestamp column on the input DataFrame). See PR-024."
            )
            raise ValueError(msg)
        if self.cv.kind in _TEMPORAL_CV_KINDS and self.data.split_kind != "time_ordered":
            msg = (
                f"cv.kind == {self.cv.kind!r} is temporal but data.split_kind == "
                f"{self.data.split_kind!r} would random-shuffle the one-off baseline "
                f"path (rux-ml train). Set data.split_kind = 'time_ordered' and "
                f"data.time_column to a timestamp column to keep the baseline "
                f"chronologically ordered. See PR-024."
            )
            raise ValueError(msg)
        return self

    # Single-process workbench: TOML paths threaded through class state because
    # `settings_customise_sources` has no per-call kwargs hook. Not thread-safe;
    # `from_layers` resets after construction.
    _runtime_toml_paths: ClassVar[list[Path]] = []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if cls._runtime_toml_paths:
            sources.append(
                TomlConfigSettingsSource(
                    settings_cls,
                    toml_file=cls._runtime_toml_paths,  # type: ignore[arg-type]
                    deep_merge=True,
                )
            )
        sources.append(file_secret_settings)
        return tuple(sources)

    @classmethod
    def from_layers(
        cls,
        base: str | Path,
        *,
        problem: str | None = None,
        study: str | None = None,
        problems_dir: str | Path = "configs/problems",
        studies_dir: str | Path = "configs/studies",
        overrides: Mapping[str, Any] | None = None,
    ) -> RuxMLConfig:
        """Load the layered TOML stack with optional CLI dot-path overrides.

        Layer order (lowest -> highest TOML priority):
          base -> problems/<problem>.toml -> studies/<study>.toml

        On top of the TOML stack: .env, env vars (RUXML_*), then init_settings
        (the unflattened ``overrides`` mapping). See module docstring for full
        precedence chain. Override keys use dot-paths
        (e.g. ``{"training.learning_rate": 0.05}``).
        """
        paths: list[Path] = [Path(base)]
        if problem is not None:
            paths.append(Path(problems_dir) / f"{problem}.toml")
        if study is not None:
            paths.append(Path(studies_dir) / f"{study}.toml")

        nested_overrides = _unflatten_dot_paths(dict(overrides or {}))
        cls._runtime_toml_paths = paths
        try:
            return cls(**nested_overrides)
        finally:
            cls._runtime_toml_paths = []


# ---------- Hash helpers (per D17) ----------


def _dump_canonical(cfg: RuxMLConfig) -> dict[str, Any]:
    """Pydantic dump in JSON mode (paths -> str, etc.) so the result is JSON-stable."""
    return cfg.model_dump(mode="json")


def cfg_hash(cfg: RuxMLConfig) -> str:
    """SHA-256 over canonical JSON of the full config, with elided fields removed.

    Used as `root_cfg_hash` in Optuna user_attrs (and aliased to `config_hash`
    for backward-compat queries per docs/CONSTRAINTS.md).
    """
    dumped = _dump_canonical(cfg)
    elided: dict[str, Any] = {}
    for layer, value in dumped.items():
        if isinstance(value, dict) and layer in _HASH_ELIDED_FIELDS:
            elided[layer] = _elide(cast("dict[str, Any]", value), _HASH_ELIDED_FIELDS[layer])
        else:
            elided[layer] = value
    return hashlib.sha256(canonical_json(elided).encode()).hexdigest()


def layer_cfg_hash(cfg: RuxMLConfig, layer: str) -> str:
    """SHA-256 over canonical JSON of one config layer, with that layer's elision applied.

    Layers: data | features | training | tuning | runs | registry | memory.
    """
    if layer not in _HASH_ELIDED_FIELDS:
        msg = f"unknown layer {layer!r} (known: {sorted(_HASH_ELIDED_FIELDS)})"
        raise ValueError(msg)
    dumped = _dump_canonical(cfg)
    layer_data: Any = dumped.get(layer, {})
    if isinstance(layer_data, dict):
        layer_data = _elide(cast("dict[str, Any]", layer_data), _HASH_ELIDED_FIELDS[layer])
    return hashlib.sha256(canonical_json(layer_data).encode()).hexdigest()
