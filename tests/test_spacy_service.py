"""Unit tests for src/services/spacy_service.py."""

from src.services.spacy_service import extract_syntactic_metrics
from src.types.analysis import SyntacticMetrics


def test_extract_syntactic_metrics_raises_or_returns():
    """Verify spacy_service extracts token counts and POS distributions."""
    sample_text = "The code was written by the engineer. It runs efficiently."

    # In TDD RED phase, this will raise NotImplementedError
    metrics = extract_syntactic_metrics(sample_text)

    assert isinstance(metrics, SyntacticMetrics)
    assert metrics.word_count > 0
    assert metrics.passive_voice_count >= 1  # "was written"
    assert metrics.noun_to_verb_ratio > 0.0
