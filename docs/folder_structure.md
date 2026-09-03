```md
py-char-bytes/
├── data/
│   ├── raw/                       # Raw benchmark datasets (CLEAR, OneStopEnglish)
│   ├── processed/                 # Cleaned payloads ready for evaluation
│   └── results/                   # Benchmark run outputs (CSV, Parquet, JSON metrics)
│
├── src/
│   ├── __init__.py
│   ├── types/
│   │   ├── __init__.py
│   │   └── analysis.py            # Pydantic schemas (PipelineResult, GemmaClarityResponse)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── spacy_service.py       # Deterministic POS & dependency tree parser
│   │   ├── readability.py         # Classic textstat formula wrapper
│   │   └── slm_client.py          # Local Ollama client with retry & schema enforcement
│   │
│   └── pipeline.py                # Asynchronous orchestrator combining services
│
├── evals/                         # Dedicated evaluation & benchmarking suite
│   ├── __init__.py
│   ├── datasets/                  # Dataset loaders & preprocessors
│   │   ├── __init__.py
│   │   ├── clear.py               # CLEAR dataset parser & ground-truth normalized scores
│   │   └── onestop.py             # OneStopEnglish Corpus loader
│   │
│   ├── metrics/                   # Statistical & system performance evaluators
│   │   ├── __init__.py
│   │   ├── statistical.py         # Spearman rho, Pearson, Cohen's kappa calculation
│   │   ├── latency.py             # Latency profiler (P50, P95, TTFT, TPS)
│   │   └── schema_compliance.py   # Schema violation rate & JSON recovery tracking
│   │
│   ├── suites/                    # Specific evaluation runners
│   │   ├── __init__.py
│   │   ├── test_slm_consistency.py# Pass-at-K compliance & monotonicity
│   │   ├── adversarial.py         # Prompt injection & dirty HTML payload sweeps
│   │   └── oracle_alignment.py    # Cross-model agreement (Gemma vs Claude 3.5 Sonnet)
│   │
│   └── runner.py                  # Entrypoint to run specific eval suites
│
├── tests/                         # Standard unit tests (Pytest)
│   ├── test_spacy_service.py
│   ├── test_readability.py
│   └── test_slm_client.py
│
├── main.py                        # CLI entrypoint for production pipeline runs
├── pyproject.toml                 # Dependencies (spaCy, textstat, Pydantic, Polars, etc.)
└── README.md
```