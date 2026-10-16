"""Process-wide environment pinning + version capture (per D10 + PR-011 + PR-013).

Two surfaces live here:

- :func:`pin_threads` (PR-011 sub-decision A1) — sets the thread-pool /
  parallelism env vars the workbench means to export **before** importing
  numpy / polars / sklearn / xgboost. Those libraries read the env when they
  size their thread pools; setting them later is a no-op against the
  existing pool. **As of PR-062 that ordering holds on no path**: every
  caller below pins after xgboost (and numpy) have loaded — measured on the
  desktop, ``OMP_NUM_THREADS`` 24 set, ``omp_get_max_threads()`` 32 (program
  PR-082 R3-a; PR-062 probe 2). The XGBoost / CatBoost thread budget
  therefore travels as an explicit argument from ``cfg.memory.omp_threads``
  (``make_trainer(..., threads=...)``; the native adapter's ``memory_cfg``),
  not through this env var. PR-063 (measured on the desktop, 2026-10-16):
  libgomp and both OpenBLAS copies (numpy's, scipy's) size their pools in
  load-time constructors, so a pin after the import reaches neither; Polars
  sizes on first use and LightGBM maps its own libgomp lazily, so the pin
  still reaches both. The trial child therefore receives the four values in
  its spawn env (:func:`thread_env`, passed by
  ``tuning/isolation.py``); the in-process verbs (``train``, in-process
  ``tune``, ``solve``) still pin after the import, so their libgomp /
  OpenBLAS run at the library default (32 on the desktop) — a CLI re-exec
  was not built (PR-063 Q3).
- :func:`get_versions` (PR-013) — captures the runtime version provenance
  block that lands in ``TrialAttrs`` (``xgboost_version``,
  ``cuda_runtime_version``, ``omp_threads``, ``image_digest``,
  ``gpu_model``, ``driver_version``). ``xgboost`` is imported lazily so
  callers in the subprocess child don't trigger it before :func:`pin_threads`
  has run.

Callers:

- ``tuning/isolation.py`` — the parent spawns the trial child with
  :func:`thread_env` over its own environment (PR-063), the only route that
  reaches libgomp / OpenBLAS in the child.
- ``_internal/trial_runner.py`` — the subprocess child warns when the env it
  started with disagrees with :func:`thread_env` (PR-063), then calls
  :func:`pin_threads` after loading ``RuxMLConfig`` (PR-008 pattern,
  refactored here in PR-011); that import already loads xgboost and numpy
  (``rux_ml.config`` → ``config/training.py`` → ``rux_ml.training``), so the
  heavy stack is not lazy there (PR-062 probe 2; PR-063). Calls
  :func:`get_versions` from inside the trial body (after heavy imports).
- ``cli/train.py`` and ``cli/tune.py`` — in-process CLI verbs call
  :func:`pin_threads` at the start of the verb body so a user running
  ``rux-ml train`` directly (without subprocess isolation) still gets the
  right thread budget.
- ``runs/attrs.py`` — :meth:`TrialAttrs.from_cfg` consumes
  :func:`get_versions` to populate the environment block.

Per D10's cross-runtime limitation of ``threadpoolctl`` (libgomp vs libiomp),
env-var pinning is THE mechanism; we deliberately do NOT import
``threadpoolctl`` here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

if TYPE_CHECKING:
    from rux_ml.config import MemoryConfig


# The .docker-image-digest file is written by `make docker-build` (per PR-012).
# When the workbench runs outside the container, the file is absent and the
# digest is recorded as the literal string "unknown" — PR-010 still validates
# the provenance triple, but explicit unknown is honest about what we know.
_IMAGE_DIGEST_PATH = Path(".docker-image-digest")
_UNKNOWN = "unknown"


THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


def thread_env(memory: MemoryConfig) -> dict[str, str]:
    """The four thread-pool env values for ``memory`` (keys :data:`THREAD_ENV_KEYS`).

    The one source for :func:`pin_threads` and for the trial child's spawn env
    (PR-063): a value set before a process starts reaches the pools that size
    at load (libgomp, OpenBLAS); a value set after the import does not.
    """
    return {
        "OMP_NUM_THREADS": str(memory.omp_threads),
        "OPENBLAS_NUM_THREADS": str(memory.openblas_threads),
        "MKL_NUM_THREADS": str(memory.mkl_threads),
        "POLARS_MAX_THREADS": str(memory.polars_threads),
    }


def pin_threads(memory: MemoryConfig) -> None:
    """Export ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS``
    / ``POLARS_MAX_THREADS`` from a resolved ``MemoryConfig``.

    Idempotent: callers may invoke it multiple times. Sets the env vars
    unconditionally (overwrites any inherited shell value) so the workbench's
    intended budget always wins.
    """
    os.environ.update(thread_env(memory))


class EnvironmentVersions(NamedTuple):
    """Runtime version provenance for one trial (per CONSTRAINTS.md)."""

    xgboost_version: str
    cuda_runtime_version: str
    omp_threads: int
    image_digest: str
    gpu_model: str | None
    driver_version: str | None


def _read_image_digest() -> str:
    """Return the contents of ``.docker-image-digest`` or ``"unknown"``."""
    try:
        digest = _IMAGE_DIGEST_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return _UNKNOWN
    return digest if digest else _UNKNOWN


def _query_nvidia_smi(field: str) -> str | None:
    """Run ``nvidia-smi --query-gpu=<field>`` and return the trimmed value.

    Returns ``None`` when ``nvidia-smi`` is absent (CPU-only dev hosts) or
    when the call fails for any reason — these signals are best-effort and
    must never block a trial.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    # Multi-GPU hosts would return one line per GPU; we only care about the first.
    first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    return first_line or None


def _cuda_version_str() -> str:
    """Return ``"<major>.<minor>"`` from ``xgboost.build_info()``.

    ``xgboost.build_info()`` is the XGBoost-3.x canonical surface for build
    metadata (per the API confirmed in PR-012's container smoke output:
    ``CUDA_VERSION: [12, 9]``). The legacy ``xgboost.config_context()`` is
    for *configuring* XGBoost, not querying the build, and does not expose
    CUDA_VERSION.
    """
    import xgboost as xgb  # noqa: PLC0415 — lazy per PR-008 module-load order

    info = xgb.build_info()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    cuda = info.get("CUDA_VERSION")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    # CUDA_VERSION is documented as [major, minor]; reject anything else as unknown.
    _expected_len = 2
    if not cuda or not isinstance(cuda, list) or len(cast("list[object]", cuda)) < _expected_len:
        return _UNKNOWN
    parts = cast("list[object]", cuda)
    return f"{int(cast('int', parts[0]))}.{int(cast('int', parts[1]))}"


def get_versions(memory: MemoryConfig) -> EnvironmentVersions:
    """Capture the runtime version provenance for one trial.

    Always derivable (required on :class:`TrialAttrs`):
      - ``xgboost_version``: ``xgboost.__version__``
      - ``cuda_runtime_version``: ``xgboost.build_info()["CUDA_VERSION"]``
      - ``omp_threads``: ``memory.omp_threads`` (the pinned budget)
      - ``image_digest``: ``.docker-image-digest`` contents or ``"unknown"``

    Optional (GPU-only — populated when ``nvidia-smi`` is available):
      - ``gpu_model``: e.g. ``"NVIDIA GeForce RTX 4090"``
      - ``driver_version``: e.g. ``"580.142"``

    Callers must have pinned thread env vars via :func:`pin_threads` BEFORE
    invoking this — ``xgboost`` is imported here, and that triggers OpenMP
    pool sizing.
    """
    import xgboost as xgb  # noqa: PLC0415 — lazy per PR-008 module-load order

    return EnvironmentVersions(
        xgboost_version=xgb.__version__,
        cuda_runtime_version=_cuda_version_str(),
        omp_threads=memory.omp_threads,
        image_digest=_read_image_digest(),
        gpu_model=_query_nvidia_smi("name"),
        driver_version=_query_nvidia_smi("driver_version"),
    )
