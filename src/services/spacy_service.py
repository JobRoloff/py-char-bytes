"""Service module for deterministic syntactic parsing and dependency extraction using spaCy."""

from functools import cache

import spacy

from src.types.analysis import SyntacticMetrics


@cache
def _load_nlp():
    """Load the configured spaCy pipeline lazily so unit tests can patch it."""
    return spacy.load("en_core_web_sm")


def _zero_metrics() -> SyntacticMetrics:
    return SyntacticMetrics(
        word_count=0,
        passive_voice_count=0,
        noun_to_verb_ratio=0.0,
    )


def extract_syntactic_metrics(text: str) -> SyntacticMetrics:
    """Extract syntactic, dependency-tree, and POS distribution metrics from a text payload.

    Args:
        text: The raw text string to analyze.

    Returns:
        SyntacticMetrics containing word count, passive voice count, and noun-to-verb ratio.
    """
    if not text.strip():
        return _zero_metrics()

    doc = _load_nlp()(text)
    lexical_tokens = [token for token in doc if not token.is_punct and not token.is_space]
    passive_voice_count = sum(1 for token in doc if token.dep_ in {"auxpass", "aux:pass"})
    noun_count = sum(1 for token in lexical_tokens if token.pos_ in {"NOUN", "PROPN"})
    verb_count = sum(1 for token in lexical_tokens if token.pos_ == "VERB")
    noun_to_verb_ratio = round(noun_count / verb_count, 2) if verb_count else 0.0

    return SyntacticMetrics(
        word_count=len(lexical_tokens),
        passive_voice_count=passive_voice_count,
        noun_to_verb_ratio=noun_to_verb_ratio,
    )
