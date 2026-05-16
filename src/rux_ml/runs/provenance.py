"""Per-trial provenance helpers shared by ``cli/train.py`` and ``cli/tune.py``.

Extracted from PR-006's ``cli/train.py`` private helpers during PR-007 so both
the 1-trial baseline path and the sweep loop record the same provenance triple
in Optuna ``user_attrs``. PR-009 (run logging) will build query/compare helpers
on top of this surface.

Layered above ``data``, ``config``, and ``_internal``; consumed by ``cli/`` —
no cli imports here, per ``docs/CONVENTIONS.md`` "Module Dependency Rules".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rux_ml._internal.git import git_sha
from rux_ml.config import cfg_hash, layer_cfg_hash
from rux_ml.data import compute_data_hash

if TYPE_CHECKING:
    from rux_ml.config import RuxMLConfig

#: Per-trial config-hash layers recorded in Optuna ``user_attrs``. The order is
#: stable to keep the recorded set diff-friendly; ``cv`` was added by PR-015
#: (CV strategy joining the per-trial provenance triple per ``docs/CONSTRAINTS.md``).
HASH_LAYERS: tuple[str, ...] = (
    "data",
    "features",
    "training",
    "tuning",
    "runs",
    "registry",
    "memory",
    "cv",
)

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def study_name(cfg: RuxMLConfig, *, problem: str | None, study: str | None) -> str:
    """Substitute ``{problem}/{study}/{stamp}`` into ``cfg.runs.study_name_template``.

    Falls back to ``"default"`` / ``"oneoff"`` when ``problem`` / ``study`` are
    unset — used by ``rux-ml train`` (no ``--study``) so the one-off baseline
    still lands in a uniquely-named study.
    """
    stamp = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
    return cfg.runs.study_name_template.format(
        problem=problem or "default",
        study=study or "oneoff",
        stamp=stamp,
    )


def ensure_storage_parent(storage_url: str) -> None:
    """Create the parent dir for a ``sqlite:///…`` URL so Optuna can open it.

    No-op for non-SQLite URLs.
    """
    if not storage_url.startswith("sqlite:///"):
        return
    db_path = Path(storage_url.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)


def data_hashes(source_path: Path) -> dict[str, str]:
    """Compute the composite + per-component data hashes for ``source_path``.

    Returns:
        ``{"data_hash", "data_bytes_hash", "data_logical_hash"}``. ``data_hash``
        composes the bytes + logical hashes (matching ``snapshot()``'s version_id
        derivation) so promoted runs can cross-reference snapshots by the same
        identifier (per ``docs/ARCHITECTURE.md`` "Reproducibility Architecture").
    """
    composite = compute_data_hash(source_path)
    return {
        "data_bytes_hash": composite["bytes_hash"],
        "data_logical_hash": composite["logical_hash"],
        "data_hash": f"{composite['bytes_hash']}|{composite['logical_hash']}",
    }


def build_user_attrs(cfg: RuxMLConfig, data_hashes_: dict[str, str]) -> dict[str, str]:
    """Assemble the per-trial ``user_attrs`` set (PR-006 + PR-015 subset).

    Records the 8 per-layer ``*_cfg_hash`` fields, the root ``root_cfg_hash``,
    the ``git_sha``, and the composite + per-component data hashes. The full
    provenance triple (``entropy_hex``, ``image_digest``, library/CUDA
    versions, ``peak_rss_mb``) lands in PR-011 + PR-013.
    """
    attrs: dict[str, str] = {
        f"{layer}_cfg_hash": layer_cfg_hash(cfg, layer) for layer in HASH_LAYERS
    }
    attrs["root_cfg_hash"] = cfg_hash(cfg)
    attrs["git_sha"] = git_sha()
    attrs.update(data_hashes_)
    return attrs
