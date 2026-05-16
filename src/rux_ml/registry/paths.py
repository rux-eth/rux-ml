"""Version-id formatter + filesystem-path helpers (per PR-010 sub-decision C1).

Version-id format: ``v_{YYYY}_{MM}_{DD}_{short_hash}`` per D8 BEST-GUESS.
``short_hash`` is the first 6 chars of ``root_cfg_hash`` so the version-id is
deterministic for a given config+data and chronologically sortable.

Registry layout::

    registry/
    └── <problem>/
        ├── champion.json
        ├── v_2026_05_16_a8f3c2/
        │   ├── pipeline.skops
        │   ├── model.ubj
        │   └── manifest.json
        └── v_2026_05_17_b1e9d4/
            └── ...
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Number of hex chars from ``root_cfg_hash`` used in the version-id suffix.
SHORT_HASH_LEN: int = 6


def format_version_id(root_cfg_hash: str, *, fmt: str, now: datetime | None = None) -> str:
    """Substitute ``{date}`` and ``{short_hash}`` into ``fmt``.

    Args:
        root_cfg_hash: Hex root_cfg_hash from the originating trial.
        fmt: Format string from ``cfg.registry.version_format``.
            Default: ``"v_{date}_{short_hash}"``.
        now: Override timestamp (test seam); defaults to UTC now.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y_%m_%d")
    return fmt.format(date=stamp, short_hash=root_cfg_hash[:SHORT_HASH_LEN])


def problem_dir(registry_root: Path, problem: str) -> Path:
    """``<registry_root>/<problem>/`` — per-problem subtree root."""
    return registry_root / problem


def version_dir(registry_root: Path, problem: str, version: str) -> Path:
    """``<registry_root>/<problem>/<version>/`` — one promoted bundle's directory."""
    return registry_root / problem / version


def champion_path(registry_root: Path, problem: str) -> Path:
    """``<registry_root>/<problem>/champion.json`` — the per-problem pointer."""
    return registry_root / problem / "champion.json"
