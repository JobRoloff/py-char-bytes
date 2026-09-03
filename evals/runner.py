"""Benchmark execution runner evaluating pipeline performance against CLEAR ground truth."""

import asyncio
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# -------------------------------------------------------------------
# Path Bootstrap & Imports
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evals.datasets.clear import load_clear_dataframe
from evals.metrics import (
    calculate_ground_truth_correlations,
    calculate_latency_percentiles,
    calculate_schema_metrics,
)
from src.pipeline import process_text_payload
from src.types.analysis import PipelineResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("BenchmarkRunner")

DATA_RESULTS_DIR = BASE_DIR / "data" / "results"


async def run_clear_benchmark(sample_size: int | None = 100, max_concurrency: int = 4) -> None:
    """Execute evaluation benchmark against the CLEAR ground-truth corpus.

    Args:
        sample_size: Number of passages to sample for the run (None for full dataset).
        max_concurrency: Maximum concurrent async requests to local Ollama instance.
    """
    DATA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Sanitized Dataset
    clear_df = load_clear_dataframe()
    if sample_size and sample_size < len(clear_df):
        clear_df = clear_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {sample_size} passages for benchmark run.")

    records = clear_df.to_dict(orient="records")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def worker(record: dict) -> PipelineResult:
        async with semaphore:
            return await process_text_payload(
                doc_id=str(record["document_id"]),
                text=str(record["raw_text"]),
            )

    logger.info(f"Processing passages with max_concurrency={max_concurrency}...")
    start_time = time.perf_counter()
    tasks = [worker(rec) for rec in records]
    pipeline_results: list[PipelineResult] = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - start_time

    # 2. Extract Flattened Results
    flat_results = [r.to_flat_dict() for r in pipeline_results]
    res_df = pd.DataFrame(flat_results)

    # 3. Merge Ground Truth CLEAR Easiness Score
    merged_df = pd.merge(
        clear_df[["document_id", "clear_easiness"]],
        res_df,
        on="document_id",
        how="inner",
    )

    # 4. Calculate Modular Evaluation & Performance Metrics
    schema_stats = calculate_schema_metrics(merged_df)
    latency_stats = calculate_latency_percentiles(merged_df["latency_ms"])

    valid_mask = merged_df["schema_valid"] & merged_df["gemma_clarity_score"].notna()
    valid_df = merged_df[valid_mask] if int(valid_mask.sum()) > 2 else pd.DataFrame()

    # Calculate ground truth rank correlations using evals/metrics/statistical.py
    slm_correlations = (
        calculate_ground_truth_correlations(
            valid_df["gemma_clarity_score"], valid_df["clear_easiness"]
        )
        if not valid_df.empty
        else {}
    )

    flesch_correlations = (
        calculate_ground_truth_correlations(
            valid_df["flesch_reading_ease"], valid_df["clear_easiness"]
        )
        if not valid_df.empty
        else {}
    )

    fog_correlations = (
        calculate_ground_truth_correlations(
            valid_df["gunning_fog"], valid_df["clear_easiness"]
        )
        if not valid_df.empty
        else {}
    )

    # 5. Print Executive Benchmark Summary Report
    print("\n" + "=" * 60)
    print("      CLEAR BENCHMARK EVALUATION SUMMARY REPORT      ")
    print("=" * 60)
    print(f"Total Passages Evaluated: {schema_stats['total_evaluated']}")
    print(
        f"Schema Compliance Rate:   {schema_stats['compliance_rate_pct']:.2f}% "
        f"({schema_stats['valid_count']}/{schema_stats['total_evaluated']})"
    )
    print(f"Total Execution Duration: {total_elapsed:.2f}s")
    print("-" * 60)
    print("Latency Profile (Inference Latency ms):")
    print(f"  • P50 (Median):         {latency_stats['p50_ms']} ms")
    print(f"  • P95 Percentile:       {latency_stats['p95_ms']} ms")
    print(f"  • P99 Percentile:       {latency_stats['p99_ms']} ms")
    print("-" * 60)

    if not valid_df.empty:
        print("Ground-Truth Alignment Metrics (CLEAR BT_easiness):")
        print(
            f"  • Gemma SLM Score (Spearman ρ): {slm_correlations.get('spearman_rho', 0.0):.4f} "
            f"(p={slm_correlations.get('spearman_pvalue', 1.0):.4e})"
        )
        print(f"  • Gemma SLM Score (Pearson r):  {slm_correlations.get('pearson_r', 0.0):.4f}")
        print(
            f"  • Flesch Reading Ease (Spearman ρ): {flesch_correlations.get('spearman_rho', 0.0):.4f}"
        )
        print(f"  • Gunning Fog Index (Spearman ρ):   {fog_correlations.get('spearman_rho', 0.0):.4f}")
    else:
        print("Insufficient valid schema rows to calculate statistical alignment.")

    print("=" * 60 + "\n")

    # 6. Export Results CSV
    output_path = DATA_RESULTS_DIR / "clear_benchmark_results.csv"
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Full benchmark export saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_clear_benchmark(sample_size=20, max_concurrency=4))