"""rux-ml — personal ML research workbench for tabular GBM lifecycle.

Public API is curated and minimal per docs/CONVENTIONS.md. Most callers should
prefer qualified imports (e.g., `from rux_ml.tuning import run_tuning`).
"""

from importlib.metadata import version as _version

__version__ = _version("rux-ml")

__all__: list[str] = [
    "__version__",
]
