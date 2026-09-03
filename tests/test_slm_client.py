"""Unit tests for src/services/slm_client.py using httpx mocks."""

import pytest
import respx
from httpx import Response

from src.services.slm_client import SLMClient
from src.types.analysis import GemmaClarityResponse


@pytest.mark.asyncio
@respx.mock
async def test_slm_client_success():
    """Verify SLMClient parses valid JSON response from Ollama endpoint."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={
                "response": '{"tone": "Academic", "clarity_score": 8, "summary": "Concise overview."}'
            },
        )
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate")

    # In TDD RED phase, this will raise NotImplementedError
    response, latency_ms, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is True
    assert isinstance(response, GemmaClarityResponse)
    assert response.clarity_score == 8
    assert latency_ms > 0.0


@pytest.mark.asyncio
@respx.mock
async def test_slm_client_invalid_schema_fallback():
    """Verify SLMClient handles malformed JSON without throwing unhandled exceptions."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={"response": '{"clarity_score": 999}'},  # Invalid score > 10
        )
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate")

    response, latency_ms, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is False
    assert response is None
    assert latency_ms > 0.0
