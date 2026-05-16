"""Two-file model bundle write / read (per D8 + ``docs/CONSTRAINTS.md``).

Each bundle directory holds:

- ``pipeline.skops`` — sklearn feature ``Pipeline`` via ``skops.io``
- ``model.ubj`` — XGBoost ``Booster`` via ``Booster.save_model``
- ``manifest.json`` — Pydantic-validated provenance + library versions

Combined pickles are forbidden per ``CONSTRAINTS.md`` "Two-File Model Bundle".

``load_bundle`` uses skops' ``get_untrusted_types`` + explicit ``trusted=[…]``
since skops 0.14+ no longer accepts ``trusted=True``. For bundles we wrote
ourselves the trust list is the full set of types discovered in the file —
internal use means we accept what we just dumped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import skops.io as sio
import xgboost as xgb

from rux_ml.registry.manifest import ModelManifest, read, write

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


PIPELINE_FILENAME = "pipeline.skops"
BOOSTER_FILENAME = "model.ubj"
MANIFEST_FILENAME = "manifest.json"


def save_bundle(
    pipeline: Pipeline,
    booster: xgb.Booster,
    manifest: ModelManifest,
    dest_dir: Path,
) -> None:
    """Write the three bundle files into ``dest_dir``.

    The manifest is written atomically (tmp + ``os.replace`` per
    :func:`rux_ml.registry.manifest.write`). The pipeline and booster aren't
    rewritten in place — they're written once per version directory.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    sio.dump(pipeline, dest_dir / PIPELINE_FILENAME)
    booster.save_model(str(dest_dir / BOOSTER_FILENAME))
    write(dest_dir / MANIFEST_FILENAME, manifest)


def load_bundle(dir_path: Path) -> tuple[Pipeline, xgb.Booster, ModelManifest]:
    """Load ``(Pipeline, Booster, ModelManifest)`` from a bundle directory.

    Raises:
        FileNotFoundError: if any of the three bundle files is missing.
        ManifestSchemaError: if ``manifest.json`` fails Pydantic validation.
    """
    dir_path = Path(dir_path)
    pipeline_path = dir_path / PIPELINE_FILENAME
    booster_path = dir_path / BOOSTER_FILENAME
    manifest_path = dir_path / MANIFEST_FILENAME
    for p in (pipeline_path, booster_path, manifest_path):
        if not p.exists():
            msg = f"missing {p.name} in bundle dir {dir_path}"
            raise FileNotFoundError(msg)

    # skops 0.14+: enumerate untrusted types, then trust them explicitly.
    untrusted = sio.get_untrusted_types(file=pipeline_path)
    pipeline: Pipeline = sio.load(pipeline_path, trusted=untrusted)

    booster = xgb.Booster()
    booster.load_model(str(booster_path))  # pyright: ignore[reportUnknownMemberType]

    manifest = read(manifest_path)
    return pipeline, booster, manifest
