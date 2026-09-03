## Things to Test

services:
- types
- pipeline
- readability
- spacy nlp
- small language model client (unit)
- small language model client (model eval)
- adversarial stress testing (model eval)

### Types

At first glance a bit redundant because we're using pydantic.

However, pydantic types include field validation such as value boundaries and structural transformation.

Key checks:
- Field bounds validation (e.g., `clarity_score` strictly bounded between 1 and 10)
- Structural flattening (`to_flat_dict()`) for clean Polars/Pandas DataFrame conversion

### Pipeline

Does our pipeline asynchronously orchestrate and compute all metrics:
- Deterministic (`spaCy` syntactic metrics & `textstat` readability indices)
- Non-deterministic (Ollama generative inference)

Note the more detailed testing per metric will be done in a different file(s).

### Readability

Metrics in question:
- flesch reading ease
- gunning fog
- coleman liau
- dale chall

For each metric, are the computed values within range?

//TODO: tests to add
- Value computation correctness against known reference passages
- Edge cases handling: empty strings, single-word inputs, and special character strings

### Spacy Service

CPU-based syntactic parsing including the following:
- word count
- passive voice word count (`auxpass` dependency parsing)
- noun to verb ratio

//TODO: tests to add
- Correct identification of passive voice sentence structures
- Edge case handling: empty strings, whitespace-only input, and non-English text payloads

### Small Language Model (unit)

Focus: Software plumbing & API contract reliability (Fast, offline unit tests using `respx` mocks)

Key checks:
- Proper stripping and parsing of markdown code fences (e.g., ` ```json ` wrappers)
- Graceful error handling for out-of-bounds or malformed JSON outputs without unhandled exceptions (`schema_valid = False`)
- Resilience to network faults, HTTP 500 status codes, and request timeouts

### Small Language Model (model eval)

Focus: Non-deterministic generative behavior & semantic alignment (Live GPU execution against local Ollama instance)

Key checks:
- Is our SLM's output adhering to our desired schema across repeated runs (Pass-at-$K$ schema rate)?
- Is our SLM's output consistent under zero-temperature conditions ($\sigma^2$ variance checks)?
- Will the SLM behave as expected if input text semantics change?
- Will a semantically corrupted text output a lower clarity score - as expected (semantic monotonicity)?

### Adversarial Stress Testing (model eval)

Focus: System robustness, guardrail enforcement, and dirty payload resilience under untrusted inputs (`evals/suites/test_adversarial.py`)

Key checks:
- Does the system prompt boundary resist indirect prompt injection attempts (e.g., instructions attempting to hijack `clarity_score` or `tone`)?
- Does the pipeline safely process noisy HTML markup, unescaped quotes, and script injection tags (`<script>`) without crashing?
- Can the system handle Unicode floods, multi-language scripts, and extreme token repetitions without zero-division errors or execution hangs?

## Command Reference

uv run ruff check --fix .
uv run ruff format .
uv run pytest -m "not integration"
uv run pytest evals/suites/test_slm_consistency.py -s
uv run pytest evals/suites/test_adversarial.py -s