"""``champion.json`` atomic-rewrite helpers (per PR-010 sub-decisions D1 + E1).

The champion-pointer is a small JSON file holding the current version + audit
metadata. ``promote`` and ``rollback`` rewrite it atomically via tmp +
``os.replace`` so concurrent readers never see a partial write.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def now_iso() -> str:
    """ISO 8601 UTC timestamp with seconds precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_champion(
    path: Path,
    *,
    version: str,
    promoted_from_study: str,
    promoted_from_trial_number: int,
    metric_value: float,
    promoted_at: str | None = None,
) -> None:
    """Atomically write ``champion.json`` via tmp + ``os.replace``.

    The schema is the PR-010 sub-decision D1 shape: ``version`` + ``promoted_at``
    + ``promoted_from`` (study + trial_number + metric_value). Atomic write
    guarantees concurrent readers never see a partial file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "version": version,
        "promoted_at": promoted_at or now_iso(),
        "promoted_from": {
            "study": promoted_from_study,
            "trial_number": promoted_from_trial_number,
            "metric_value": metric_value,
        },
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_champion(path: Path) -> dict[str, Any]:
    """Load ``champion.json`` as a plain dict."""
    raw: object = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        msg = f"champion.json at {path} must contain a JSON object, got {type(raw).__name__}"
        raise TypeError(msg)
    return cast("dict[str, Any]", raw)
