"""Memory + threading config (per D10)."""

from rux_ml.config._strict_model import StrictModel


class MemoryConfig(StrictModel):
    # psutil watchdog (BEST-GUESS threshold per D10: 36 - 4 OS - 4 Docker = 28 GB).
    watchdog_threshold_gb: float = 28.0
    watchdog_sample_hz: float = 1.0

    # Thread allocation per D10 — sequential trials, each library uses all cores.
    omp_threads: int = 24
    openblas_threads: int = 1
    mkl_threads: int = 1
    polars_threads: int = 24
