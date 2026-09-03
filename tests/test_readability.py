"""Unit tests for src/services/readability.py."""

from src.services.readability import extract_readability_metrics
from src.types.analysis import ReadabilityMetrics


def test_extract_readability_metrics():
    """Verify readability service returns populated formula metrics."""
    sample_text = (
        "Simple sentences are easy to read. Complex technical documentation "
        "containing multi-syllabic vocabulary increases grade level difficulty."
    )

    # In TDD RED phase, this will raise NotImplementedError
    metrics = extract_readability_metrics(sample_text)

    assert isinstance(metrics, ReadabilityMetrics)
    assert 0.0 <= metrics.flesch_reading_ease <= 100.0
    assert metrics.gunning_fog > 0.0
    assert metrics.coleman_liau_index > 0.0
    assert metrics.dale_chall_score > 0.0
