"""Filesystem model registry — promoted bundles, atomic champion pointer, thin loader.

Per D8 + PR-010 sub-decision B1: the **public surface here is intentionally
inference-only**. ``load_model`` and the schema types are exported; the
promote / rollback code lives in :mod:`rux_ml.registry.promote` and is NOT
re-exported. Inference clients can ``from rux_ml.registry import load_model``
without dragging in the training stack (``rux_ml.tuning``, ``rux_ml.data.cv``,
``rux_ml.training``).

The "Inference-deps test" in ``tests/registry/`` enforces this separation:
it imports ``rux_ml.registry`` in a subprocess with the training-stack
sub-packages blocked in ``sys.modules`` and verifies a model still loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rux_ml.registry.bundle import load_bundle
from rux_ml.registry.champion import read_champion
from rux_ml.registry.manifest import (
    LibraryVersions,
    ManifestSchemaError,
    ModelManifest,
    PromotedFrom,
)
from rux_ml.registry.paths import champion_path, version_dir

if TYPE_CHECKING:
    import xgboost as xgb
    from sklearn.pipeline import Pipeline


def load_model(
    problem: str,
    *,
    version: str = "champion",
    registry_root: Path = Path("registry"),
) -> tuple[Pipeline, xgb.Booster]:
    """Load a promoted model bundle as ``(Pipeline, Booster)``.

    Args:
        problem: Per-problem subtree (``registry/<problem>/``).
        version: Either ``"champion"`` (default; reads ``champion.json``) or an
            explicit version-id like ``"v_2026_05_16_a8f3c2"``.
        registry_root: Filesystem root (default ``Path("registry")``).

    Returns:
        ``(Pipeline, Booster)`` — the manifest is NOT returned here so the
        inference surface stays minimal. Callers needing the manifest use
        :func:`rux_ml.registry.bundle.load_bundle` directly.
    """
    registry_root = Path(registry_root)
    if version == "champion":
        champ = read_champion(champion_path(registry_root, problem))
        version_str = str(champ["version"])
    else:
        version_str = version
    pipeline, booster, _manifest = load_bundle(version_dir(registry_root, problem, version_str))
    return pipeline, booster


__all__ = [
    "LibraryVersions",
    "ManifestSchemaError",
    "ModelManifest",
    "PromotedFrom",
    "load_model",
]
