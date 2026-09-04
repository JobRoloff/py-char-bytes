"""Service module for computing classical deterministic readability formulas via textstat."""

from textstat import textstat

from src.types.analysis import ReadabilityMetrics


def _zero_metrics() -> ReadabilityMetrics:
    return ReadabilityMetrics(
        flesch_reading_ease=0.0,
        gunning_fog=0.0,
        coleman_liau_index=0.0,
        dale_chall_score=0.0,
    )


def extract_readability_metrics(text: str) -> ReadabilityMetrics:
    """Compute classical deterministic readability scores (Flesch, Gunning Fog, etc.).

    Args:
        text: The raw text string to analyze.

    Returns:
        ReadabilityMetrics object populated with numeric readability indices.
    """
    if not text.strip() or not any(char.isalnum() for char in text):
        return _zero_metrics()

    return ReadabilityMetrics(
        flesch_reading_ease=textstat.flesch_reading_ease(text),
        gunning_fog=textstat.gunning_fog(text),
        coleman_liau_index=textstat.coleman_liau_index(text),
        dale_chall_score=textstat.dale_chall_readability_score(text),
    )
