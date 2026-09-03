"""Unit tests for src/services/readability.py."""

import math

import pytest

from src.services.readability import extract_readability_metrics
from src.types.analysis import ReadabilityMetrics


def test_extract_readability_metrics():
    """Verify readability service returns populated formula metrics."""
    sample_text = (
        "Simple sentences are easy to read. Complex technical documentation "
        "containing multi-syllabic vocabulary increases grade level difficulty."
    )

    metrics = extract_readability_metrics(sample_text)

    assert isinstance(metrics, ReadabilityMetrics)
    assert math.isfinite(metrics.flesch_reading_ease)
    assert metrics.gunning_fog > 0.0
    assert metrics.coleman_liau_index > 0.0
    assert metrics.dale_chall_score > 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "The cat sat on the mat. It was sunny.",
            {
                "flesch_reading_ease": 108.2675,
                "gunning_fog": 1.8,
                "coleman_liau_index": -4.977777777777776,
                "dale_chall_score": 0.2232,
            },
        ),
        (
            (
                "Photosynthesis converts light energy into chemical energy through "
                "chlorophyll-mediated reactions."
            ),
            {
                "flesch_reading_ease": -48.655,
                "gunning_fog": 28.0,
                "coleman_liau_index": 31.12,
                "dale_chall_score": 15.1855,
            },
        ),
    ],
)
def test_extract_readability_metrics_matches_textstat_reference_values(text, expected):
    """Lock formula outputs against known textstat examples, including unbounded Flesch."""
    metrics = extract_readability_metrics(text)

    assert metrics.flesch_reading_ease == pytest.approx(expected["flesch_reading_ease"])
    assert metrics.gunning_fog == pytest.approx(expected["gunning_fog"])
    assert metrics.coleman_liau_index == pytest.approx(expected["coleman_liau_index"])
    assert metrics.dale_chall_score == pytest.approx(expected["dale_chall_score"])


@pytest.mark.parametrize("text", ["", "   ", "!!! ###"])
def test_extract_readability_metrics_empty_or_non_word_payloads_return_zeroes(text):
    """Empty, whitespace, and symbol-only inputs should not produce noisy formula artifacts."""
    metrics = extract_readability_metrics(text)

    assert metrics == ReadabilityMetrics(
        flesch_reading_ease=0.0,
        gunning_fog=0.0,
        coleman_liau_index=0.0,
        dale_chall_score=0.0,
    )


def test_extract_readability_metrics_single_word_payload():
    """Single-token text is a valid edge case and should follow textstat outputs."""
    metrics = extract_readability_metrics("Word.")

    assert metrics.flesch_reading_ease == pytest.approx(121.22)
    assert metrics.gunning_fog == pytest.approx(0.4)
    assert metrics.coleman_liau_index == pytest.approx(-22.2)
    assert metrics.dale_chall_score == pytest.approx(0.0496)
