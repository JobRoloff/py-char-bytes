"""Service module for computing classical deterministic readability formulas via textstat."""

from src.types.analysis import ReadabilityMetrics


def extract_readability_metrics(text: str) -> ReadabilityMetrics:
    """Compute classical deterministic readability scores (Flesch, Gunning Fog, etc.).

    Args:
        text: The raw text string to analyze.

    Returns:
        ReadabilityMetrics object populated with numeric readability indices.

    Raises:
        NotImplementedError: Pending implementation in Phase 4.
    """
    raise NotImplementedError("extract_readability_metrics is not yet implemented.")
