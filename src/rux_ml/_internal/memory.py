"""Per-trial memory watchdog (per D10 + PR-011 sub-decisions C1/E1).

The :class:`Watchdog` context manager runs a background thread that polls
``psutil.Process(os.getpid()).memory_info().rss`` at ``cfg.memory.watchdog_sample_hz``
(default 1 Hz), tracks the peak observed RSS, and sets a ``tripped`` flag if
RSS exceeds ``cfg.memory.watchdog_threshold_gb``.

**The watchdog is observational + post-fit-check, not preemptive.** XGBoost
training is a long-running C call; Python signal/interrupt machinery from a
background thread isn't reliable inside C extensions. Callers check
``wd.tripped`` after the fit returns; if set, they raise
:class:`MemoryPressureError` which the objective wrapper converts to
``optuna.TrialPruned``.

Hard OOM kills (process exceeds OS / cgroup limit) are NOT this watchdog's
job — PR-008's subprocess isolation handles them via non-zero exit codes
and ``isolation._mark_trial_failed``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import psutil


class MemoryPressureError(Exception):
    """Raised by trial bodies when the watchdog's threshold was exceeded.

    The objective wrapper catches this and converts to ``optuna.TrialPruned``
    so the trial is recorded as ``PRUNED`` (not ``FAIL``) in storage.
    """


_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * 1024 * 1024


@dataclass
class _WatchdogState:
    """Shared state between the trial body and the sampling thread.

    ``stop`` is the public-by-convention name (no leading underscore) because
    the sampling thread + ``Watchdog.__exit__`` both access it — basedpyright's
    ``reportPrivateUsage`` flags cross-class private access.
    """

    threshold_bytes: int
    sample_interval_s: float
    peak_bytes: int = 0
    tripped: bool = False
    stop: threading.Event = field(default_factory=threading.Event)


def _sample_loop(state: _WatchdogState, proc: psutil.Process) -> None:
    """Background poller: every ``sample_interval_s``, read RSS + update peak."""
    while not state.stop.is_set():
        try:
            rss = proc.memory_info().rss
        except psutil.NoSuchProcess:  # parent process gone; abort sampling
            return
        state.peak_bytes = max(state.peak_bytes, rss)
        if rss > state.threshold_bytes:
            state.tripped = True
        # Wait or exit early when stopped.
        state.stop.wait(state.sample_interval_s)


class Watchdog:
    """Context manager wrapping a trial body with the RSS-watching background thread.

    Usage::

        with Watchdog(threshold_gb=cfg.memory.watchdog_threshold_gb,
                      sample_hz=cfg.memory.watchdog_sample_hz) as wd:
            ... do per-trial work ...

        # After exit, inspect peak / tripped state for provenance + pruning.
        if wd.tripped:
            raise MemoryPressureError(...)
        record_peak_rss_mb(wd.peak_mb)
    """

    def __init__(self, *, threshold_gb: float, sample_hz: float) -> None:
        self._state = _WatchdogState(
            threshold_bytes=int(threshold_gb * _BYTES_PER_GB),
            sample_interval_s=1.0 / max(sample_hz, 1e-6),
        )
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())

    def __enter__(self) -> Watchdog:
        # Seed peak with current RSS so even pruned-before-first-sample trials get a value.
        self._state.peak_bytes = self._proc.memory_info().rss
        self._thread = threading.Thread(
            target=_sample_loop, args=(self._state, self._proc), daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._state.stop.set()
        # Tiny wait so the final RSS sample lands; daemon=True will reap if it overruns.
        if self._thread is not None:
            self._thread.join(timeout=2.0 * self._state.sample_interval_s)
        # One last RSS read so the recorded peak reflects post-fit memory if higher.
        try:
            final_rss = self._proc.memory_info().rss
            self._state.peak_bytes = max(self._state.peak_bytes, final_rss)
        except psutil.NoSuchProcess:
            pass

    @property
    def peak_mb(self) -> float:
        """Peak observed RSS in MB. Always set after `__enter__` (seeded from current RSS)."""
        return self._state.peak_bytes / _BYTES_PER_MB

    @property
    def tripped(self) -> bool:
        """``True`` if any sample exceeded ``threshold_gb`` during the wrapped block."""
        return self._state.tripped

    def force_trip(self) -> None:
        """Test seam — simulate a threshold trip without actually allocating memory."""
        self._state.tripped = True
