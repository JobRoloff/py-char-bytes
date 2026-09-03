"""Unit tests for Pydantic schema validation and bounds checking in src/types/analysis.py."""

import pytest
from pydantic import ValidationError

from src.types.analysis import (
    DeterministicAnalysis,
    GemmaClarityResponse,
    PipelineResult,
    ReadabilityMetrics,
    SyntacticMetrics,
    SystemMetrics,
)


def test_gemma_clarity_response_valid():
    """Verify GemmaClarityResponse accepts valid numeric bounds."""
    data = {"tone": "Academic", "clarity_score": 8, "summary": "Clear thesis."}
    response = GemmaClarityResponse(**data)
    assert response.clarity_score == 8
    assert response.tone == "Academic"


def test_gemma_clarity_response_out_of_bounds():
    """Verify GemmaClarityResponse rejects clarity_score outside [1, 10]."""
    with pytest.raises(ValidationError):
        GemmaClarityResponse(tone="Informal", clarity_score=11, summary="Too high")

    with pytest.raises(ValidationError):
        GemmaClarityResponse(tone="Informal", clarity_score=0, summary="Too low")


def test_pipeline_result_flat_dict():
    """Verify PipelineResult.to_flat_dict flattens nested models cleanly."""
    res = PipelineResult(
        document_id="doc_123",
        deterministic=DeterministicAnalysis(
            readability=ReadabilityMetrics(
                flesch_reading_ease=70.5,
                gunning_fog=8.2,
                coleman_liau_index=7.1,
                dale_chall_score=6.4,
            ),
            syntactic=SyntacticMetrics(
                word_count=100,
                passive_voice_count=2,
                noun_to_verb_ratio=1.5,
            ),
        ),
        slm_insights=GemmaClarityResponse(
            tone="Technical",
            clarity_score=9,
            summary="Well structured.",
        ),
        system=SystemMetrics(latency_ms=142.5, schema_valid=True),
    )

    flat = res.to_flat_dict()
    assert flat["document_id"] == "doc_123"
    assert flat["flesch_reading_ease"] == 70.5
    assert flat["gemma_clarity_score"] == 9
    assert flat["latency_ms"] == 142.5
