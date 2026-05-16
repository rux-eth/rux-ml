"""Tests for ``rux_ml.registry.champion`` atomic rewrite (per PR-010 sub-decision E1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rux_ml.registry.champion import read_champion, write_champion


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "champion.json"
    write_champion(
        path,
        version="v_2026_05_16_a8f3c2",
        promoted_from_study="s1",
        promoted_from_trial_number=3,
        metric_value=0.91,
    )
    data = read_champion(path)
    assert data["version"] == "v_2026_05_16_a8f3c2"
    assert data["promoted_from"]["study"] == "s1"
    assert data["promoted_from"]["trial_number"] == 3
    assert data["promoted_from"]["metric_value"] == 0.91
    assert "promoted_at" in data


def test_atomic_write_no_partial_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os  # noqa: PLC0415

    path = tmp_path / "champion.json"
    # Seed the existing champion.
    write_champion(
        path,
        version="v_baseline",
        promoted_from_study="s0",
        promoted_from_trial_number=0,
        metric_value=0.5,
    )
    before = path.read_text()

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated rename")

    monkeypatch.setattr(os, "replace", _raise)
    with pytest.raises(OSError, match="simulated"):
        write_champion(
            path,
            version="v_new",
            promoted_from_study="s1",
            promoted_from_trial_number=1,
            metric_value=0.9,
        )

    # Concurrent readers see the pre-failure file unchanged.
    assert path.read_text() == before


def test_read_champion_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "champion.json"
    path.write_text('[1, 2, 3]')
    with pytest.raises(TypeError, match="JSON object"):
        read_champion(path)
