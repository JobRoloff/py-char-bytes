"""Integration tests for src/pipeline.py."""

import pytest
import respx
from httpx import Response

from src.pipeline import process_text_payload
from src.types.analysis import PipelineResult


@pytest.mark.asyncio
@respx.mock
async def test_process_text_payload_integration():
    """Verify pipeline integrates deterministic parsing, readability, and SLM execution."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={
                "response": '{"tone": "Informal", "clarity_score": 7, "summary": "Direct explanation."}'
            },
        )
    )

    sample_text = "Automated readability metrics evaluate text structure rapidly."

    # In TDD RED phase, this will raise NotImplementedError
    result = await process_text_payload(doc_id="test_001", text=sample_text)

    assert isinstance(result, PipelineResult)
    assert result.document_id == "test_001"
    assert result.deterministic.syntactic.word_count > 0
    assert result.deterministic.readability.flesch_reading_ease > 0.0
    assert result.slm_insights is not None
    assert result.slm_insights.clarity_score == 7
    assert result.system.schema_valid is True
