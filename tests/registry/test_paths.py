"""Tests for ``rux_ml.registry.paths`` (per PR-010 sub-decision C1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rux_ml.registry.paths import (
    SHORT_HASH_LEN,
    champion_path,
    format_version_id,
    problem_dir,
    version_dir,
)


def test_format_version_id_substitutes_date_and_short_hash() -> None:
    now = datetime(2026, 5, 16, tzinfo=UTC)
    vid = format_version_id("a8f3c2deadbeef", fmt="v_{date}_{short_hash}", now=now)
    assert vid == "v_2026_05_16_a8f3c2"


def test_format_version_id_short_hash_length() -> None:
    """``short_hash`` is the first SHORT_HASH_LEN chars of root_cfg_hash."""
    now = datetime(2026, 5, 16, tzinfo=UTC)
    full_hash = "0123456789abcdef" * 4
    vid = format_version_id(full_hash, fmt="v_{date}_{short_hash}", now=now)
    assert vid.endswith("_" + full_hash[:SHORT_HASH_LEN])


def test_format_version_id_alternate_format() -> None:
    """Per CONVENTIONS — `version_format` is configurable; format string drives output."""
    now = datetime(2026, 5, 16, tzinfo=UTC)
    vid = format_version_id("aabbcc", fmt="{short_hash}-{date}", now=now)
    assert vid == "aabbcc-2026_05_16"


def test_path_helpers() -> None:
    root = Path("/tmp/registry")
    assert problem_dir(root, "churn") == root / "churn"
    assert version_dir(root, "churn", "v1") == root / "churn" / "v1"
    assert champion_path(root, "churn") == root / "churn" / "champion.json"
