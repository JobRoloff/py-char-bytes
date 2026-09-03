"""Latency profiling and system duration benchmarking metrics."""

import numpy as np
import pandas as pd


def calculate_latency_percentiles(latencies_ms: pd.Series | np.ndarray | list[float]) -> dict[str, float]:
    """Calculate P50, P90, P95, and P99 latency percentiles in milliseconds.

    Args:
        latencies_ms: List or Series of document processing durations in milliseconds.

    Returns:
        dict containing mean, p50, p90, p95, and p99 latency metrics.
    """
    clean_latencies = np.asarray(latencies_ms)[pd.notna(latencies_ms)]

    if len(clean_latencies) == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    return {
        "mean_ms": round(float(np.mean(clean_latencies)), 2),
        "p50_ms": round(float(np.percentile(clean_latencies, 50)), 2),
        "p90_ms": round(float(np.percentile(clean_latencies, 90)), 2),
        "p95_ms": round(float(np.percentile(clean_latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(clean_latencies, 99)), 2),
    }