"""Data layer config (per D3, D9, D14)."""

from pathlib import Path

from pydantic import Field

from rux_ml.config._strict_model import StrictModel


class DataConfig(StrictModel):
    # Set per problem; absent in base.toml on purpose
    source_path: Path | None = None
    target_column: str | None = None

    # Content-addressed cache + manifests (per D9 + D14 Option A)
    cas_root: Path = Path("data/cas")
    manifests_root: Path = Path("data/manifests")

    # XGBoost ingest decision-rule threshold (per D3 ARCHITECTURE.md "Decision Rules")
    # X estimated below this size -> QuantileDMatrix on GPU; above -> ExtMemQuantileDMatrix.
    gpu_in_memory_x_gb_max: float = 18.0

    # Train / val / test ratios (sum should be 1.0)
    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
