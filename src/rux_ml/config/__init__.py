"""Config package — Pydantic-settings models composed into RuxMLConfig.

See `docs/ARCHITECTURE.md` "Configuration Architecture" and D2 / D16 / D17.
"""

from rux_ml.config.data import DataConfig
from rux_ml.config.features import FeaturesConfig
from rux_ml.config.memory import MemoryConfig
from rux_ml.config.registry import RegistryConfig
from rux_ml.config.root import RuxMLConfig, cfg_hash, layer_cfg_hash
from rux_ml.config.runs import RunsConfig
from rux_ml.config.training import TrainingConfig
from rux_ml.config.tuning import CatSpec, FloatSpec, IntSpec, SearchSpec, TuningConfig

__all__ = [
    "CatSpec",
    "DataConfig",
    "FeaturesConfig",
    "FloatSpec",
    "IntSpec",
    "MemoryConfig",
    "RegistryConfig",
    "RunsConfig",
    "RuxMLConfig",
    "SearchSpec",
    "TrainingConfig",
    "TuningConfig",
    "cfg_hash",
    "layer_cfg_hash",
]
