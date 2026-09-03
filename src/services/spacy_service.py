"""Service module for deterministic syntactic parsing and dependency extraction using spaCy."""

from src.types.analysis import SyntacticMetrics


def extract_syntactic_metrics(text: str) -> SyntacticMetrics:
    """Extract syntactic, dependency-tree, and POS distribution metrics from a text payload.

    Args:
        text: The raw text string to analyze.

    Returns:
        SyntacticMetrics containing word count, passive voice count, and noun-to-verb ratio.

    Raises:
        NotImplementedError: Pending implementation in Phase 4.
    """
    raise NotImplementedError("extract_syntactic_metrics is not yet implemented.")
