"""Unit tests for src/services/slm_client.py using httpx mocks."""

import httpx
import pytest
import respx
from httpx import Response

from src.services.slm_client import SLMClient
from src.types.analysis import GemmaClarityResponse


@pytest.mark.asyncio
@respx.mock
async def test_slm_client_success():
    """Verify SLMClient parses clean, valid JSON response from Ollama endpoint."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={
                "response": '{"tone": "Academic", "clarity_score": 8, "summary": "Concise overview."}'
            },
        )
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate")
    response, latency_ms, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is True
    assert isinstance(response, GemmaClarityResponse)
    assert response.clarity_score == 8
    assert latency_ms > 0.0


@pytest.mark.asyncio
@respx.mock
async def test_slm_client_markdown_code_fence_handling():
    """Verify SLMClient strips markdown code fences (```json) before parsing."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={
                "response": '```json\n{\n  "tone": "Informal",\n  "clarity_score": 7,\n  "summary": "Fenced output."\n}\n```'
            },
        )
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate")
    response, _, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is True
    assert isinstance(response, GemmaClarityResponse)
    assert response.clarity_score == 7
    assert response.tone == "Informal"


@pytest.mark.asyncio
@respx.mock
async def test_slm_client_invalid_schema_fallback():
    """Verify SLMClient handles out-of-bounds schema values without crashing."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={"response": '{"tone": "Academic", "clarity_score": 999, "summary": "Invalid"}'},
        )
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate")
    response, latency_ms, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is False
    assert response is None
    assert latency_ms > 0.0


@pytest.mark.asyncio
@respx.mock
async def test_slm_client_timeout_handling():
    """Verify SLMClient catches HTTP timeouts gracefully."""
    respx.post("http://localhost:11434/api/generate").mock(
        side_effect=httpx.TimeoutException("Connection timed out")
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate", timeout=1.0)
    response, latency_ms, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is False
    assert response is None
    assert latency_ms > 0.0