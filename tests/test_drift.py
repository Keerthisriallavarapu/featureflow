"""Tests for drift detection."""
from __future__ import annotations

import numpy as np

from featureflow.monitoring import population_stability_index


def test_psi_no_drift_when_distributions_match():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 10_000)
    current = rng.normal(0, 1, 10_000)
    result = population_stability_index(baseline, current)
    assert result.method == "psi"
    assert result.statistic < 0.1, f"Expected low PSI, got {result.statistic}"
    assert result.drifted is False


def test_psi_detects_mean_shift():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 10_000)
    current = rng.normal(2, 1, 10_000)  # mean shifted by 2 sigma
    result = population_stability_index(baseline, current)
    assert result.statistic > 0.5, f"Expected high PSI, got {result.statistic}"
    assert result.drifted is True


def test_psi_handles_empty_arrays():
    result = population_stability_index(np.array([]), np.array([1, 2, 3]))
    assert result.statistic == 0.0
    assert result.drifted is False


def test_psi_threshold_can_be_overridden():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 10_000)
    current = rng.normal(0.3, 1, 10_000)  # moderate shift
    relaxed = population_stability_index(baseline, current, threshold=0.5)
    strict = population_stability_index(baseline, current, threshold=0.05)
    assert relaxed.drifted is False or relaxed.statistic >= 0.5
    assert strict.drifted is True
