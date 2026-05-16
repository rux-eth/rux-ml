"""Runs layer config (per D7)."""

from pathlib import Path

from rux_ml.config._strict_model import StrictModel


class RunsConfig(StrictModel):
    # Optuna SQLite storage URL (per D6 + D7).
    storage_url: str = "sqlite:///studies/studies.db"

    # FileSystemArtifactStore root (per D7).
    artifacts_root: Path = Path("studies/artifacts")

    # Naming pattern for new studies; substitution applied by tuning.study.
    study_name_template: str = "{problem}_{study}_{stamp}"
