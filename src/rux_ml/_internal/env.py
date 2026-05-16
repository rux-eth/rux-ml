"""Process-wide environment pinning (per D10 + PR-011 sub-decision A1).

Single canonical surface for the thread-pool / parallelism env vars the
workbench must set **before** importing numpy / polars / sklearn / xgboost.
Those libraries read the env at import-time to size their thread pools;
setting them later is a no-op against the existing pool.

Callers:

- ``_internal/trial_runner.py`` — the subprocess child calls
  :func:`pin_threads` after loading ``RuxMLConfig`` and before lazy-importing
  the heavy stack (PR-008 pattern, refactored here in PR-011).
- ``cli/train.py`` and ``cli/tune.py`` — in-process CLI verbs call
  :func:`pin_threads` at the start of the verb body so a user running
  ``rux-ml train`` directly (without subprocess isolation) still gets the
  right thread budget.

Per D10's cross-runtime limitation of ``threadpoolctl`` (libgomp vs libiomp),
env-var pinning is THE mechanism; we deliberately do NOT import
``threadpoolctl`` here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rux_ml.config import MemoryConfig


def pin_threads(memory: MemoryConfig) -> None:
    """Export ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS``
    / ``POLARS_MAX_THREADS`` from a resolved ``MemoryConfig``.

    Idempotent: callers may invoke it multiple times. Sets the env vars
    unconditionally (overwrites any inherited shell value) so the workbench's
    intended budget always wins.
    """
    os.environ["OMP_NUM_THREADS"] = str(memory.omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(memory.openblas_threads)
    os.environ["MKL_NUM_THREADS"] = str(memory.mkl_threads)
    os.environ["POLARS_MAX_THREADS"] = str(memory.polars_threads)
