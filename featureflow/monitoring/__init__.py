"""Drift detection for features and predictions.

Two flavors implemented:
- Population Stability Index (PSI) for numeric features. Standard in
  financial/credit modeling. Bins values and compares distributions.
- Chi-square for categorical.

Both compare a baseline distribution (the training data) against a recent
window of production data. Values above thresholds trigger alerts.

We keep these as pure functions rather than a 'DriftDetector' class because
the state (baseline) is owned elsewhere — usually loaded from the model
artifact at deploy time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DriftResult:
    statistic: float
    threshold: float
    drifted: bool
    method: str
    bins: int | None = None


# Rule of thumb thresholds:
# PSI < 0.1 = no significant change
# 0.1 <= PSI < 0.25 = some shift, investigate
# PSI >= 0.25 = significant drift, alert
PSI_DEFAULT_THRESHOLD = 0.25


def population_stability_index(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    threshold: float = PSI_DEFAULT_THRESHOLD,
) -> DriftResult:
    """PSI between two numeric arrays.

    PSI = sum over bins of (current_pct - baseline_pct) * ln(current_pct / baseline_pct)

    Small epsilon added to avoid log(0).
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    if baseline.size == 0 or current.size == 0:
        return DriftResult(0.0, threshold, False, "psi", bins)

    # Use baseline quantiles as bin edges so bins are balanced
    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        # Degenerate baseline (all values equal); fall back to range bins
        edges = np.linspace(baseline.min(), baseline.max() + 1e-9, bins + 1)

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    baseline_pct = baseline_counts / max(baseline_counts.sum(), 1) + eps
    current_pct = current_counts / max(current_counts.sum(), 1) + eps

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    return DriftResult(
        statistic=psi,
        threshold=threshold,
        drifted=psi >= threshold,
        method="psi",
        bins=len(edges) - 1,
    )


def categorical_drift(
    baseline: list[str],
    current: list[str],
    threshold: float = 0.05,
) -> DriftResult:
    """Chi-square test for categorical drift. Returns p-value as statistic;
    drifted = p_value < threshold (default 0.05).

    Note: chi-square assumes expected count >= 5 per cell; for low-count
    categories you'd want Fisher's exact instead. We log a warning.
    """
    from scipy import stats  # local import; scipy is in `ml` extra

    cats = sorted(set(baseline) | set(current))
    baseline_counts = np.array([baseline.count(c) for c in cats])
    current_counts = np.array([current.count(c) for c in cats])

    # Build observed/expected for chi-square
    total = baseline_counts.sum() + current_counts.sum()
    expected_baseline = (baseline_counts + current_counts) * baseline_counts.sum() / total
    expected_current = (baseline_counts + current_counts) * current_counts.sum() / total

    if np.any(expected_baseline < 5) or np.any(expected_current < 5):
        log.warning("Low expected counts; chi-square unreliable. Consider Fisher's exact.")

    observed = np.array([baseline_counts, current_counts])
    expected = np.array([expected_baseline, expected_current])
    chi2 = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-9)))
    dof = max(len(cats) - 1, 1)
    p_value = 1.0 - float(stats.chi2.cdf(chi2, dof))

    return DriftResult(
        statistic=p_value,
        threshold=threshold,
        drifted=p_value < threshold,
        method="chi_square",
    )
