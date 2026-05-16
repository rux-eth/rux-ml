"""Meta-tests on the golden tolerance helpers (per PR-014 verification criterion).

Validates that :func:`assert_predictions_close` and :func:`assert_metric_within`
actually reject a deliberate perturbation — they're the regression gate, and
a gate that's a no-op is worse than no gate at all (false sense of safety).
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    DEFAULT_ATOL,
    DEFAULT_METRIC_BAND,
    assert_metric_within,
    assert_predictions_close,
)

pytestmark = pytest.mark.golden


def test_assert_predictions_close_accepts_within_tolerance() -> None:
    """Sanity: identical inputs pass."""
    golden = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    actual = golden.copy()
    assert_predictions_close(actual, golden)


def test_assert_predictions_close_rejects_perturbation_above_atol() -> None:
    """A deliberate perturbation of ``10 * atol`` raises AssertionError."""
    golden = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    actual = golden + (10 * DEFAULT_ATOL)
    with pytest.raises(AssertionError, match="diverged from golden_v1"):
        assert_predictions_close(actual, golden)


def test_assert_predictions_close_rejects_shape_mismatch() -> None:
    golden = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    actual = np.array([0.1, 0.5], dtype=np.float64)
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_predictions_close(actual, golden)


def test_assert_metric_within_accepts_inside_band() -> None:
    baseline = 0.875
    assert_metric_within(baseline + (DEFAULT_METRIC_BAND / 2), baseline)
    assert_metric_within(baseline - (DEFAULT_METRIC_BAND / 2), baseline)


def test_assert_metric_within_rejects_outside_band() -> None:
    baseline = 0.875
    drifted = baseline + (10 * DEFAULT_METRIC_BAND)
    with pytest.raises(AssertionError, match="drifted beyond"):
        assert_metric_within(drifted, baseline)
