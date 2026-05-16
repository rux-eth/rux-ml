"""Conftest for the golden-regression suite (per PR-014).

Wires:
- ``--regenerate-golden`` pytest CLI flag (per :func:`pytest_addoption`) so
  ``make regenerate-golden`` can refresh ``tests/golden/fixtures/golden_v1/``.
  When the flag is set, the golden tests compute fresh fixtures and write
  them; when not set (default), they assert against the committed fixtures.
- Tolerance helpers :func:`assert_predictions_close` (atol=1e-5, rtol=1e-4)
  and :func:`assert_metric_within` (AUC abs_tol=0.005). Both wrap
  ``np.testing`` calls with golden-specific diff messages so a CUDA / XGBoost
  upgrade failure points at the right investigation procedure.
- Autouse ``OMP_NUM_THREADS=1`` fixture so the golden CPU fit is bit-exact
  per the PR-013 determinism contract; the spec's `atol=1e-5/rtol=1e-4` is
  the conservative envelope across XGBoost minor versions, not the bit-exact
  floor.

Per ``docs/CONSTRAINTS.md`` Tolerance-Based Golden Tests Only: never
``np.testing.assert_array_equal`` against committed fixtures — XGBoost GPU
``hist`` is not bit-exact across hardware, and the goldens are designed to
survive a CUDA / XGBoost minor upgrade.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

# Repo-relative path to the committed golden fixtures.
GOLDEN_DIR: Path = Path(__file__).resolve().parent / "fixtures" / "golden_v1"

# Tolerance defaults — locked-in PR-014 sub-decision E1.
DEFAULT_ATOL: float = 1e-5
DEFAULT_RTOL: float = 1e-4
DEFAULT_METRIC_BAND: float = 0.005


# ---------------------------------------------------------------------------
# Pytest CLI flag
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--regenerate-golden`` to the pytest CLI surface.

    Used by ``make regenerate-golden`` to refresh ``preds.npy`` / ``metric.json``
    / ``manifest.json`` after a deliberate library upgrade. Default behavior
    (flag NOT set) compares against the committed fixtures.
    """
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help=(
            "Regenerate the committed golden fixtures in "
            "tests/golden/fixtures/golden_v1/ instead of asserting against "
            "them. Intended for manual use via `make regenerate-golden`; "
            "NEVER run in CI."
        ),
    )


@pytest.fixture
def regenerate_golden(request: pytest.FixtureRequest) -> bool:
    """Expose ``--regenerate-golden`` to tests as a boolean fixture."""
    return bool(request.config.getoption("--regenerate-golden"))


# ---------------------------------------------------------------------------
# Determinism env-var pinning (CPU bit-exact contract per PR-013)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pin_omp_single_thread(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU bit-exact requires single-threaded OpenMP — pin for this module.

    Mirrors ``tests/integration/test_determinism_cpu.py`` so the golden fit
    runs under the same conditions PR-013's determinism contract was
    verified against. basedpyright doesn't track ``@pytest.fixture(autouse=True)``
    usage; the inline pyright ignore on the def line suppresses that.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------


def assert_predictions_close(
    actual: NDArray[np.float64],
    golden: NDArray[np.float64],
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """Wrap :func:`np.testing.assert_allclose` with golden-specific guidance.

    Raises ``AssertionError`` when ``actual`` diverges from ``golden`` beyond
    the tolerance envelope. The message points the investigator at the
    documented procedure in the test docstring rather than at the immediate
    fix (regenerating).
    """
    if actual.shape != golden.shape:
        msg = (
            f"predictions shape mismatch: actual={actual.shape} vs "
            f"golden={golden.shape}. The fixture was generated for a "
            f"different dataset shape — investigate before regenerating."
        )
        raise AssertionError(msg)
    try:
        np.testing.assert_allclose(actual, golden, atol=atol, rtol=rtol)
    except AssertionError as exc:
        max_abs = float(np.max(np.abs(actual - golden)))
        max_rel_denom = float(np.max(np.abs(golden)))
        max_rel = max_abs / max_rel_denom if max_rel_denom > 0 else float("inf")
        msg = (
            f"predictions diverged from golden_v1 beyond atol={atol}, rtol={rtol}.\n"
            f"  max absolute diff: {max_abs:.6g}\n"
            f"  max relative diff: {max_rel:.6g}\n"
            f"Investigation procedure (per PR-014):\n"
            f"  1. Diff `manifest.json` library versions vs current "
            f"`xgboost.__version__` + cuda_runtime_version.\n"
            f"  2. Run `uv run pytest tests/integration/test_determinism_cpu.py` "
            f"to confirm the PR-013 determinism contract still holds.\n"
            f"  3. Only after BOTH check out, consider `make regenerate-golden` "
            f"and commit the refreshed fixtures with a manual diff review."
        )
        raise AssertionError(msg) from exc


def assert_metric_within(
    metric_value: float,
    baseline: float,
    *,
    abs_tol: float = DEFAULT_METRIC_BAND,
) -> None:
    """Assert a scalar metric (AUC, RMSE, etc.) lies within ``abs_tol`` of ``baseline``.

    Raises ``AssertionError`` with the absolute deviation when the band is
    exceeded.
    """
    diff = abs(metric_value - baseline)
    if diff > abs_tol:
        msg = (
            f"metric drifted beyond ±{abs_tol} band: actual={metric_value:.6f}, "
            f"baseline={baseline:.6f}, |diff|={diff:.6f}. Same investigation "
            f"procedure as `assert_predictions_close`."
        )
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Fixed-seed synthetic data fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden_synthetic_path() -> Iterator[Path]:
    """Yield the path to the committed ``synthetic.parquet`` golden input.

    The fixture is not regenerated here — the file is committed and any
    regeneration is the responsibility of ``--regenerate-golden`` paths in
    the golden tests themselves (which use the same deterministic
    construction documented in ``test_xgb_baseline.py``).
    """
    parquet_path = GOLDEN_DIR / "synthetic.parquet"
    assert parquet_path.exists(), (
        f"missing committed golden input at {parquet_path}; "
        f"if you're bootstrapping the fixtures, run "
        f"`uv run pytest -m golden --regenerate-golden`"
    )
    yield parquet_path
