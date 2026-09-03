# Benchmark Experiment: Hybrid Computational NLP and Local SLM Evaluation Pipeline

## 1. Abstract
Automated text evaluation in production pipelines requires balancing deterministic precision, qualitative semantic reasoning, schema reliability, and computational efficiency. Traditional readability metrics (e.g., Flesch-Kincaid, Gunning Fog) provide ultra-fast, zero-variance syntactic signals but lack semantic context. Conversely, Small Language Models (SLMs) offer rich qualitative judgment but introduce non-deterministic execution, latent runtime delays, and structural drift.

This benchmark evaluates an enterprise-grade hybrid analytics pipeline integrating deterministic syntactic parsing (**spaCy**), classic readability algorithms (**textstat**), and local generative inference (**Ollama running Gemma:2b**). Rather than relying solely on high-level LLM-as-a-Judge ratings, this framework measures **multi-dimensional system reliability**: evaluating rank correlation against human-annotated ground truth (CLEAR dataset), schema violation rates, inference performance ($P_{50}/P_{95}$ latency, tokens/sec), and robustness under dirty or adversarial input payloads.

---

## 2. Experimental Objectives

* **Production Architecture:** Engineer an end-to-end asynchronous Python pipeline converting raw text payloads into structured quantitative metrics and dynamic LLM evaluations with failure-tolerant error handling.
* **Multi-Layered Feature Extraction:**
  * **Deterministic Syntactic Engine:** Extract structural token ratios, dependency-tree patterns (e.g., passive auxiliary counts), and POS distributions via **spaCy**.
  * **Classical Readability Indexing:** Compute Flesch Reading Ease, Gunning Fog, Coleman-Liau, and Dale-Chall scores via **textstat**.
  * **Structured SLM Insights:** Enforce structured JSON generation from a locally hosted `gemma:2b` model via strict **Pydantic** schema definitions, capturing qualitative tone and clarity ratings ($1 \dots 10$).
* **Comprehensive Evaluation Matrix:**
  * **Human Ground-Truth Alignment:** Validate SLM and classical score alignment against educator-annotated readability corpora (**CLEAR**, **OneStopEnglish**).
  * **Inter-Model Agreement & Oracle Drift:** Benchmark local `gemma:2b` outputs against Claude 3.5 Sonnet / GPT-4o oracle evaluations using Cohen’s $\kappa$ and Spearman’s $\rho$.
  * **System Performance & Latency:** Profile Time to First Token (TTFT), Tokens Per Second (TPS), and $P_{50}/P_{95}/P_{99}$ latency distributions across parameter sweeps ($T \in [0.0, 0.7]$).
  * **Schema Reliability & Edge-Case Resilience:** Stress-test the pipeline against invalid JSON outputs, prompt injection attacks, extreme sentence lengths, and HTML/noisy payloads.

---

## 3. Evaluation Taxonomy & Benchmark Metrics

### A. Deterministic Syntactic & Readability Baselines
* **Flesch Reading Ease:** Surface-level structural check measuring sentence length and syllable density ($0\dots100$).
* **Gunning Fog Index:** Evaluates grade-level readability based on complex word percentages ($\ge 3$ syllables).
* **Coleman-Liau Index:** Efficient character-level formula, avoiding syllable lookup overhead in high-throughput applications.
* **Dale-Chall Readability:** Uses a 3,000-word reference vocabulary lookup; sensitive to domain-specific jargon regardless of word length.

### B. Ground-Truth Datasets & Oracle Benchmarks
* **CLEAR Dataset (CommonLit Ease of Readability):** 4,700+ educator-labeled passages providing continuous reading difficulty scores ($\mu=0, \sigma=1$).
* **OneStopEnglish Corpus:** Parallel news articles across Elementary, Intermediate, and Advanced tiers for categorical alignment testing.
* **Oracle Judge Baseline:** Sampling subsets through Claude 3.5 Sonnet and GPT-4o to establish inter-annotator agreement (Cohen’s $\kappa$) and quantify local model performance bounds.

### C. System Engineering & Reliability Metrics
* **Schema Adherence Rate (%):** Percentage of raw generative outputs matching the required JSON schema without falling back to regex or repair tools.
* **Latency Profile ($P_{50}, P_{95}$):** End-to-end execution duration per document and Time To First Token (TTFT).
* **Inference Efficiency (TPS):** Generated output tokens per second on local hardware resources.
* **Adversarial Pass Rate (%):** Percentage of runs where system prompts resist indirect prompt injections embedded inside payload text.

---

## 4. Pipeline Data Schema

| Field Name | Type | Source | System / Validation | Description |
| :--- | :--- | :--- | :--- | :--- |
| `document_id` | `str` | Input | UUID | Unique identifier |
| `raw_text` | `str` | Input | Sanitized Text | Input payload under analysis |
| `word_count` | `int` | spaCy | $\text{len}(\text{doc})$ | Total token count |
| `passive_voice_count` | `int` | spaCy | `dep_ == "auxpass"` | Count of passive auxiliary constructions |
| `noun_to_verb_ratio` | `float` | spaCy | POS Analysis | Structural nominalization density |
| `flesch_reading_ease`| `float` | `textstat` | Deterministic Algorithm | Standard readability index ($0 \dots 100$) |
| `gunning_fog` | `float` | `textstat` | Deterministic Algorithm | Complex-word grade level rating |
| `coleman_liau_index` | `float` | `textstat` | Deterministic Algorithm | Character-based grade level index |
| `dale_chall_score` | `float` | `textstat` | Lexicon Lookup | Vocabulary difficulty rating |
| `gemma_clarity_score`| `int` | Gemma:2b | Pydantic Field ($1 \dots 10$) | Structured rating of text flow |
| `gemma_tone` | `str` | Gemma:2b | Pydantic Enum | Classified document tone |
| `latency_ms` | `float` | Pipeline | System Benchmark | End-to-end Ollama inference duration |
| `schema_valid` | `bool` | Pipeline | Validation Check | Flag indicating strict JSON schema match |

---

## 5. Testing

The project has two testing surfaces:

- **Unit tests** live under `tests/` and are the default fast feedback loop.
- **Evaluation suites** live under `evals/suites/` and are opt-in because they require a live local Ollama runtime.

Run the default unit test suite:

```bash
uv run pytest
```

During the current TDD red phase, this command is expected to fail on the unimplemented service and pipeline stubs. To run only the already-green schema and benchmark-helper tests:

```bash
uv run pytest tests/test_types.py tests/test_eval_metrics.py tests/test_clear_dataset.py
```

Run linting and formatting checks:

```bash
uv run ruff check .
uv run ruff format .
```

Run the live SLM consistency evals after starting Ollama with `gemma:2b` available:

```bash
ollama run gemma:2b
uv run pytest evals/suites/test_slm_consistency.py -s
```

Run the adversarial pipeline evals:

```bash
uv run pytest evals/suites/test_adversarial.py -s
```

The live eval suites are marked `integration`; keep them separate from the unit suite unless you are intentionally validating local model behavior.

## 6. Production Pipeline Implementation Code

```python
import json
import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import requests
from scipy.stats import spearmanr
import spacy
import textstat

# -------------------------------------------------------------------
# Logging & Setup Configuration
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EvalPipeline")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    raise

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# -------------------------------------------------------------------
# Pydantic Schemas for Strict Output Enforcement
# -------------------------------------------------------------------
class GemmaClarityResponse(BaseModel):
    tone: str = Field(..., description="Primary tone of the passage (e.g., Academic, Informal, Technical)")
    clarity_score: int = Field(..., ge=1, le=10, description="Clarity rating strictly between 1 and 10")
    summary: str = Field(..., description="Brief 1-sentence summary of core argument")

class PipelineResult(BaseModel):
    document_id: str
    word_count: int
    passive_voice_count: int
    noun_to_verb_ratio: float
    flesch_reading_ease: float
    gunning_fog: float
    coleman_liau_index: float
    dale_chall_score: float
    gemma_clarity_score: Optional[int] = None
    gemma_tone: Optional[str] = None
    gemma_summary: Optional[str] = None
    latency_ms: float = 0.0
    schema_valid: bool = False

# -------------------------------------------------------------------
# Feature Extraction Modules
# -------------------------------------------------------------------
def extract_spacy_features(text: str) -> Dict[str, Any]:
    """Extract syntactic and dependency features via spaCy."""
    doc = nlp(text)
    passive_count = sum(1 for token in doc if token.dep_ == "auxpass")
    verbs = sum(1 for t in doc if t.pos_ == "VERB")
    nouns = sum(1 for t in doc if t.pos_ == "NOUN")
    ratio = (nouns / verbs) if verbs > 0 else 0.0

    return {
        "word_count": len(doc),
        "passive_voice_count": passive_count,
        "noun_to_verb_ratio": round(ratio, 2),
    }

def extract_readability_metrics(text: str) -> Dict[str, float]:
    """Extract deterministic classical readability scores."""
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "dale_chall_score": textstat.dale_chall_readability_score(text),
    }

def extract_gemma_insights(text: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Query Ollama local model with strict schema parsing and execution profiling."""
    prompt = f"""
    Analyze the following text for clarity and structure.
    Return ONLY a JSON object matching this schema:
    {{
      "tone": "string",
      "clarity_score": integer (1 to 10),
      "summary": "string"
    }}

    Text payload:
    {text}
    """
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        raw_output = response.json().get("response", "{}")
        parsed_json = json.loads(raw_output)

        # Validate structured output against Pydantic schema
        validated = GemmaClarityResponse(**parsed_json)
        return {
            "gemma_clarity_score": validated.clarity_score,
            "gemma_tone": validated.tone,
            "gemma_summary": validated.summary,
            "latency_ms": round(elapsed_ms, 2),
            "schema_valid": True,
        }

    except (requests.RequestException, json.JSONDecodeError, ValidationError) as err:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning(f"Inference failure or schema violation: {str(err)}")
        return {
            "gemma_clarity_score": None,
            "gemma_tone": "Error",
            "gemma_summary": str(err),
            "latency_ms": round(elapsed_ms, 2),
            "schema_valid": False,
        }

# -------------------------------------------------------------------
# Benchmark Execution Engine
# -------------------------------------------------------------------
def run_benchmark(input_csv: str, output_csv: str) -> None:
    logger.info(f"Loading input payload from {input_csv}...")
    df = pd.read_csv(input_csv)

    results: List[PipelineResult] = []

    for idx, row in df.iterrows():
        doc_id = str(row.get("document_id", f"doc_{idx}"))
        text = str(row.get("raw_text", ""))

        if not text.strip():
            continue

        spacy_feats = extract_spacy_features(text)
        readability_feats = extract_readability_metrics(text)
        gemma_feats = extract_gemma_insights(text)

        result_item = PipelineResult(
            document_id=doc_id,
            **spacy_feats,
            **readability_feats,
            **gemma_feats,
        )
        results.append(result_item)

    # Convert results list to DataFrame
    res_df = pd.DataFrame([r.dict() for r in results])

    # Compute Operational and Statistical Engineering Metrics
    total_evals = len(res_df)
    valid_schemas = res_df["schema_valid"].sum()
    schema
