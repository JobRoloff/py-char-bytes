+-----------------------------+
                         |    Input Layer (Ingestion)  |
                         |  Markdown / CSV Raw Data    |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         |    Deterministic Pipeline   |
                         |   (spaCy + textstat API)    |
                         | - Syntactic Tree Parsing    |
                         | - Readability Indexes       |
                         +--------------+--------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
     +---------------------------+             +---------------------------+
     |   Local Async Engine      |             |   Oracle Sampler (10%)    |
     | (Ollama / Gemma:2b API)   |             | (Claude 3.5 / GPT-4o API) |
     | - Fast Zero-Cost Batches  |             | - Gold Standard Baseline  |
     +-------------+-------------+             +-------------+-------------+
                   |                                         |
                   +--------------------+--------------------+
                                        |
                                        v
                         +-----------------------------+
                         |  Pydantic Schema Guardrail  |
                         | - Type Validation & Clamping|
                         | - Auto-Retry / Fallbacks    |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         |     Evaluation Engine       |
                         | - MAE & Spearman (ρ) Stats  |
                         | - CI/CD Quality Gate        |
                         +-----------------------------+