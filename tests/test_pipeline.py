"""Integration tests for src/pipeline.py."""

import pytest
import respx
from httpx import Response

import src.pipeline as pipeline
from src.types.analysis import PipelineResult, ReadabilityMetrics, SyntacticMetrics


class ExplodingSLMClient:
    async def analyze_clarity(self, text: str):
        raise AssertionError("SLM should not be called for blank text")


@pytest.fixture
def deterministic_service_mocks(monkeypatch):
    def fake_syntactic_metrics(text: str) -> SyntacticMetrics:
        if not text.strip():
            return SyntacticMetrics(
                word_count=0,
                passive_voice_count=0,
                noun_to_verb_ratio=0.0,
            )
        return SyntacticMetrics(
            word_count=8,
            passive_voice_count=0,
            noun_to_verb_ratio=1.5,
        )

    def fake_readability_metrics(text: str) -> ReadabilityMetrics:
        if not text.strip():
            return ReadabilityMetrics(
                flesch_reading_ease=0.0,
                gunning_fog=0.0,
                coleman_liau_index=0.0,
                dale_chall_score=0.0,
            )
        return ReadabilityMetrics(
            flesch_reading_ease=72.5,
            gunning_fog=8.1,
            coleman_liau_index=7.4,
            dale_chall_score=6.2,
        )

    monkeypatch.setattr(
        pipeline,
        "extract_syntactic_metrics",
        fake_syntactic_metrics,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "extract_readability_metrics",
        fake_readability_metrics,
        raising=False,
    )


@pytest.mark.asyncio
@respx.mock
async def test_process_text_payload_integration(deterministic_service_mocks):
    """Verify pipeline integrates deterministic parsing, readability, and SLM execution."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={
                "response": (
                    '{"tone": "Informal", "clarity_score": 7, "summary": "Direct explanation."}'
                )
            },
        )
    )

    sample_text = "Automated readability metrics evaluate text structure rapidly."

    result = await pipeline.process_text_payload(doc_id="test_001", text=sample_text)

    assert isinstance(result, PipelineResult)
    assert result.document_id == "test_001"
    assert result.deterministic.syntactic.word_count > 0
    assert result.deterministic.readability.flesch_reading_ease > 0.0
    assert result.slm_insights is not None
    assert result.slm_insights.clarity_score == 7
    assert result.system.schema_valid is True


@pytest.mark.asyncio
@respx.mock
async def test_process_text_payload_preserves_deterministic_metrics_when_slm_schema_fails(
    deterministic_service_mocks,
):
    """Pipeline should degrade gracefully when the local model returns invalid schema."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={"response": '{"tone": "Academic", "clarity_score": 999, "summary": "Invalid"}'},
        )
    )

    result = await pipeline.process_text_payload(
        doc_id="slm_bad_schema",
        text="Readable text should still receive deterministic metrics.",
    )

    assert isinstance(result, PipelineResult)
    assert result.document_id == "slm_bad_schema"
    assert result.deterministic.syntactic.word_count > 0
    assert result.deterministic.readability.gunning_fog >= 0.0
    assert result.slm_insights is None
    assert result.system.schema_valid is False


@pytest.mark.asyncio
async def test_process_text_payload_blank_text_returns_empty_result_without_slm_call(
    deterministic_service_mocks,
):
    """Blank input should be a cheap no-op result with zero deterministic metrics."""
    result = await pipeline.process_text_payload(
        doc_id="blank",
        text="   ",
        slm_client=ExplodingSLMClient(),
    )

    assert isinstance(result, PipelineResult)
    assert result.document_id == "blank"
    assert result.deterministic.syntactic.word_count == 0
    assert result.deterministic.syntactic.passive_voice_count == 0
    assert result.deterministic.syntactic.noun_to_verb_ratio == 0.0
    assert result.deterministic.readability.flesch_reading_ease == 0.0
    assert result.deterministic.readability.gunning_fog == 0.0
    assert result.deterministic.readability.coleman_liau_index == 0.0
    assert result.deterministic.readability.dale_chall_score == 0.0
    assert result.slm_insights is None
    assert result.system.schema_valid is False
