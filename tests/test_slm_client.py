"""Unit tests for src/services/slm_client.py using httpx mocks."""

import json

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
    route = respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={
                "response": (
                    '{"tone": "Academic", "clarity_score": 8, "summary": "Concise overview."}'
                )
            },
        )
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate")
    response, latency_ms, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is True
    assert isinstance(response, GemmaClarityResponse)
    assert response.clarity_score == 8
    assert latency_ms > 0.0

    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == "gemma:2b"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 0.0
    assert "Test payload" in payload["prompt"]


@pytest.mark.asyncio
@respx.mock
async def test_slm_client_markdown_code_fence_handling():
    """Verify SLMClient strips markdown code fences (```json) before parsing."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={
                "response": (
                    '```json\n{\n  "tone": "Informal",\n  "clarity_score": 7,\n'
                    '  "summary": "Fenced output."\n}\n```'
                )
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


@pytest.mark.asyncio
@respx.mock
async def test_slm_client_http_500_fallback():
    """Verify HTTP error responses are returned as invalid schemas without exceptions."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(500, json={"error": "model unavailable"})
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate")
    response, latency_ms, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is False
    assert response is None
    assert latency_ms > 0.0


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "ollama_json",
    [
        {"response": "not valid json"},
        {},
        {"response": "[1, 2, 3]"},
        {
            "response": (
                'Here is the JSON:\n{"tone": "Academic", "clarity_score": 8, '
                '"summary": "Wrapped in prose."}'
            )
        },
    ],
)
async def test_slm_client_rejects_malformed_missing_or_non_object_response(ollama_json):
    """Only strict object JSON or fenced strict object JSON should validate."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(200, json=ollama_json)
    )

    client = SLMClient(endpoint_url="http://localhost:11434/api/generate")
    response, latency_ms, schema_valid = await client.analyze_clarity("Test payload")

    assert schema_valid is False
    assert response is None
    assert latency_ms > 0.0
