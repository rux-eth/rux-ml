"""Tests for ``rux_ml._internal.memory.Watchdog`` (per PR-011 sub-decisions C1/E1)."""

from __future__ import annotations

import time

import pytest

from rux_ml._internal.memory import MemoryPressureError, Watchdog


def test_peak_mb_is_seeded_in_enter() -> None:
    """``peak_mb`` is non-zero immediately after entering — seeded from current RSS."""
    with Watchdog(threshold_gb=999.0, sample_hz=1.0) as wd:
        # No allocation done yet; peak is just the entry-time RSS.
        assert wd.peak_mb > 0
    # Post-exit peak is still observable.
    assert wd.peak_mb > 0


def test_tripped_false_below_threshold() -> None:
    """A trivial block with a high threshold never trips the watchdog."""
    with Watchdog(threshold_gb=999.0, sample_hz=10.0) as wd:
        time.sleep(0.05)
    assert wd.tripped is False


def test_force_trip_test_seam() -> None:
    """``force_trip()`` flips the flag without actually allocating memory."""
    with Watchdog(threshold_gb=999.0, sample_hz=1.0) as wd:
        wd.force_trip()
    assert wd.tripped is True


def test_threshold_trip_against_seeded_peak() -> None:
    """Threshold tighter than the current process RSS trips the watchdog."""
    # Set threshold to a value below typical process RSS (1 MB GB ⇒ 0.001 GB);
    # any sample tick will exceed it.
    with Watchdog(threshold_gb=0.001, sample_hz=20.0) as wd:
        time.sleep(0.2)
    assert wd.tripped is True
    assert wd.peak_mb > 1  # at least 1 MB observed


def test_memory_pressure_error_is_exception() -> None:
    """``MemoryPressureError`` is an ``Exception`` subclass so handlers catch it."""
    assert issubclass(MemoryPressureError, Exception)
    with pytest.raises(MemoryPressureError, match="testing"):
        raise MemoryPressureError("testing")


def test_watchdog_records_peak_after_block_exits() -> None:
    """After the with-block exits, peak_mb is still readable."""
    with Watchdog(threshold_gb=999.0, sample_hz=10.0) as wd:
        peak_during = wd.peak_mb
        time.sleep(0.1)
    peak_after = wd.peak_mb
    # peak_after is the max-of-during-and-final-sample; never less than peak_during.
    assert peak_after >= peak_during
