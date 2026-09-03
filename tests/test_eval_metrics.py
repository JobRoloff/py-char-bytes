"""Unit tests for benchmark metric helpers in evals/metrics."""

import numpy as np
import pandas as pd

from evals.metrics.latency import calculate_latency_percentiles
from evals.metrics.schema_compliance import calculate_schema_metrics
from evals.metrics.statistical import calculate_ground_truth_correlations, calculate_oracle_kappa


def test_calculate_latency_percentiles_filters_missing_values():
    metrics = calculate_latency_percentiles([100.0, 200.0, np.nan, 300.0])

    assert metrics == {
        "mean_ms": 200.0,
        "p50_ms": 200.0,
        "p90_ms": 280.0,
        "p95_ms": 290.0,
        "p99_ms": 298.0,
    }


def test_calculate_latency_percentiles_empty_input_returns_zeroes():
    metrics = calculate_latency_percentiles([])

    assert metrics == {
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p90_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
    }


def test_calculate_schema_metrics_counts_valid_invalid_and_empty_results():
    metrics = calculate_schema_metrics(pd.DataFrame({"schema_valid": [True, False, True]}))

    assert metrics == {
        "total_evaluated": 3,
        "valid_count": 2,
        "invalid_count": 1,
        "compliance_rate_pct": 66.67,
    }

    assert calculate_schema_metrics(pd.DataFrame()) == {
        "total_evaluated": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "compliance_rate_pct": 0.0,
    }


def test_calculate_ground_truth_correlations_handles_missing_and_monotonic_values():
    metrics = calculate_ground_truth_correlations(
        pd.Series([1.0, 2.0, np.nan, 4.0]),
        pd.Series([10.0, 20.0, 30.0, 40.0]),
    )

    assert metrics["spearman_rho"] == 1.0
    assert metrics["pearson_r"] == 1.0


def test_calculate_ground_truth_correlations_requires_at_least_three_pairs():
    metrics = calculate_ground_truth_correlations(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))

    assert metrics == {
        "spearman_rho": 0.0,
        "spearman_pvalue": 1.0,
        "pearson_r": 0.0,
        "pearson_pvalue": 1.0,
    }


def test_calculate_oracle_kappa_handles_identical_and_insufficient_ratings():
    assert calculate_oracle_kappa(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])) == 1.0
    assert calculate_oracle_kappa(pd.Series([1]), pd.Series([1])) == 0.0
