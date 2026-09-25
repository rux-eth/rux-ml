"""Per-trial provenance helpers shared by ``cli/train.py`` and ``cli/tune.py``.

PR-007 extracted these from ``cli/train.py``'s private helpers; PR-009 then
formalized the recording surface as :class:`rux_ml.runs.attrs.TrialAttrs`
(the canonical write path is now ``TrialAttrs.from_cfg(cfg, hashes).record(trial)``).

This module retains the **shared utilities** that don't depend on the
``TrialAttrs`` schema itself:

- ``HASH_LAYERS`` — the ordered tuple of per-layer hash names recorded in
  every trial's ``user_attrs`` (consumed by :class:`TrialAttrs.from_cfg`).
- ``data_hashes(source_path, *, oracle)`` — composite + per-component dataset hashes
  (oracle quarantine first, per PR-040).
- ``ensure_storage_parent(url)`` — SQLite parent-dir bootstrap.
- ``study_name(cfg, problem, study)`` — template substitution.

Layered above ``data``, ``config``, and ``_internal``; consumed by ``cli/`` —
no cli imports here, per ``docs/CONVENTIONS.md`` "Module Dependency Rules".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rux_ml.data import compute_data_hash

if TYPE_CHECKING:
    from rux_ml.config import RuxMLConfig
    from rux_ml.config.data import OracleQuarantineConfig

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


def data_hashes(source_path: Path, *, oracle: OracleQuarantineConfig | None) -> dict[str, str]:
    """Compute the composite + per-component data hashes for ``source_path``.

    Returns:
        ``{"data_hash", "data_bytes_hash", "data_logical_hash"}``. ``data_hash``
        composes the bytes + logical hashes (matching ``snapshot()``'s version_id
        derivation) so promoted runs can cross-reference snapshots by the same
        identifier (per ``docs/ARCHITECTURE.md`` "Reproducibility Architecture").
    """
    composite = compute_data_hash(source_path, oracle=oracle)
    return {
        "data_bytes_hash": composite["bytes_hash"],
        "data_logical_hash": composite["logical_hash"],
        "data_hash": f"{composite['bytes_hash']}|{composite['logical_hash']}",
    }


