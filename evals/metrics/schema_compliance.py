"""Schema reliability and formatting validation metrics."""

import pandas as pd


def calculate_schema_metrics(df_results: pd.DataFrame) -> dict[str, float]:
    """Calculate Pass-at-K schema compliance rates across pipeline execution results.

    Args:
        df_results: DataFrame output from pipeline benchmark run containing 'schema_valid' field.

    Returns:
        dict containing total_evaluated, valid_count, invalid_count, and compliance_rate_pct.
    """
    total = len(df_results)
    if total == 0:
        return {
            "total_evaluated": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "compliance_rate_pct": 0.0,
        }

    valid_count = int(df_results["schema_valid"].sum())
    invalid_count = total - valid_count
    compliance_rate = (valid_count / total) * 100.0

    return {
        "total_evaluated": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "compliance_rate_pct": round(compliance_rate, 2),
    }
