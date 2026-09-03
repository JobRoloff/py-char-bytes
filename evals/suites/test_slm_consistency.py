"""Live integration evaluation suite testing Ollama / Gemma:2b for schema consistency,

score variance, and semantic monotonicity.

Prerequisites:
    Ollama must be running locally with the gemma:2b model loaded:
    $ ollama run gemma:2b
"""

import asyncio

import numpy as np
import pytest

from src.services.slm_client import SLMClient

# Apply custom markers so these slow live tests can be filtered out during fast CI runs
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture(scope="module")
def slm_client() -> SLMClient:
    """Instantiate SLMClient configured for the default local Ollama endpoint."""
    return SLMClient(endpoint_url="http://localhost:11434/api/generate", timeout=30.0)


async def test_gemma_live_endpoint_reachability(slm_client: SLMClient):
    """Sanity check to confirm the local Ollama daemon is reachable before running suite."""
    sample_text = "Testing endpoint reachability."
    response, latency_ms, schema_valid = await slm_client.analyze_clarity(sample_text)

    assert schema_valid is True, "Local Ollama daemon failed to return a valid schema."
    assert response is not None
    assert latency_ms > 0.0, "Latency execution duration should be greater than 0ms."


async def test_gemma_schema_consistency_and_variance(slm_client: SLMClient):
    """Run K=5 repeated evaluations on an identical text payload at T=0.0.

    Evaluates two system properties:
    1. Pass-at-K Schema Rate: Asserts 100% schema compliance across repetitions.
    2. Zero-Temperature Variance: Asserts score variance (sigma^2) stays below 0.5.
    """
    sample_text = (
        "Deterministic syntactic metrics provide zero-variance baselines for structural "
        "readability analysis, while local language models offer qualitative reasoning."
    )

    iterations = 5
    schema_valid_flags = []
    scores = []
    latencies = []

    for idx in range(iterations):
        response, latency_ms, schema_valid = await slm_client.analyze_clarity(sample_text)
        schema_valid_flags.append(schema_valid)
        latencies.append(latency_ms)

        if response and schema_valid:
            scores.append(response.clarity_score)

        # Brief sleep between dispatches to allow local GPU memory flushing
        await asyncio.sleep(0.1)

    # 1. Assert 100% Schema Adherence Rate
    valid_count = sum(schema_valid_flags)
    compliance_rate = (valid_count / iterations) * 100
    assert compliance_rate == 100.0, (
        f"Schema compliance dropped to {compliance_rate:.1f}% ({valid_count}/{iterations})."
    )

    # 2. Assert Low Score Variance under zero-temperature
    score_variance = float(np.var(scores))
    assert score_variance < 0.5, (
        f"High non-deterministic score variance detected at T=0.0: "
        f"var={score_variance:.2f}, scores={scores}"
    )


async def test_gemma_semantic_monotonicity_ordering(slm_client: SLMClient):
    """Verify that the SLM responds predictably to deliberate semantic corruption.

    Asserts that a pristine, well-structured technical passage earns a strictly
    higher or equal clarity score than a randomized word-salad version of the same topic.
    """
    pristine_text = (
        "The asynchronous pipeline coordinates deterministic syntactic parsing and "
        "local language model inference to output structured JSON evaluation metrics."
    )

    corrupted_text = (
        "pipeline asynchronous local syntactic model language evaluation JSON output "
        "parsing structured metrics deterministic inference coordinates."
    )

    pristine_resp, _, pristine_valid = await slm_client.analyze_clarity(pristine_text)
    corrupted_resp, _, corrupted_valid = await slm_client.analyze_clarity(corrupted_text)

    assert pristine_valid is True, "Pristine text parsing failed schema validation."
    assert corrupted_valid is True, "Corrupted text parsing failed schema validation."

    assert pristine_resp is not None
    assert corrupted_resp is not None

    print(
        f"\nMonotonicity Check -> Pristine Score: {pristine_resp.clarity_score} | "
        f"Corrupted Score: {corrupted_resp.clarity_score}"
    )

    # Assert Monotonic Order Check
    assert pristine_resp.clarity_score > corrupted_resp.clarity_score, (
        f"SLM failed monotonicity check: Pristine score ({pristine_resp.clarity_score}) "
        f"was not strictly higher than Corrupted score ({corrupted_resp.clarity_score})."
    )
