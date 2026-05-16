"""Registry layer config (per D8)."""

from pathlib import Path

from rux_ml.config._strict_model import StrictModel


class RegistryConfig(StrictModel):
    # Filesystem registry root (per D8).
    root: Path = Path("registry")

    # Version-string format (BEST-GUESS per D8). Tokens: {date}, {short_hash}.
    version_format: str = "v_{date}_{short_hash}"
