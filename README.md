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

