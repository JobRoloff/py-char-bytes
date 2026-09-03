"""Unit tests for src/services/spacy_service.py."""

from dataclasses import dataclass

import pytest

import src.services.spacy_service as spacy_service
from src.services.spacy_service import extract_syntactic_metrics
from src.types.analysis import SyntacticMetrics


@dataclass(frozen=True)
class FakeToken:
    text: str
    pos_: str
    dep_: str = ""
    is_punct: bool = False
    is_space: bool = False


class FakeNLP:
    def __init__(self, docs_by_text):
        self.docs_by_text = docs_by_text

    def __call__(self, text):
        return self.docs_by_text[text]


@pytest.fixture
def fake_nlp(monkeypatch):
    docs_by_text = {
        "The code was written by the engineer. It runs efficiently.": [
            FakeToken("The", "DET"),
            FakeToken("code", "NOUN"),
            FakeToken("was", "AUX", "auxpass"),
            FakeToken("written", "VERB"),
            FakeToken("by", "ADP"),
            FakeToken("the", "DET"),
            FakeToken("engineer", "NOUN"),
            FakeToken(".", "PUNCT", is_punct=True),
            FakeToken("It", "PRON"),
            FakeToken("runs", "VERB"),
            FakeToken("efficiently", "ADV"),
            FakeToken(".", "PUNCT", is_punct=True),
        ],
        "The engineer wrote the code. The test passes.": [
            FakeToken("The", "DET"),
            FakeToken("engineer", "NOUN"),
            FakeToken("wrote", "VERB"),
            FakeToken("the", "DET"),
            FakeToken("code", "NOUN"),
            FakeToken(".", "PUNCT", is_punct=True),
            FakeToken("The", "DET"),
            FakeToken("test", "NOUN"),
            FakeToken("passes", "VERB"),
            FakeToken(".", "PUNCT", is_punct=True),
        ],
        "漢字 العربية Русский text": [
            FakeToken("漢字", "X"),
            FakeToken("العربية", "X"),
            FakeToken("Русский", "X"),
            FakeToken("text", "NOUN"),
        ],
        "nouns only payload": [
            FakeToken("nouns", "NOUN"),
            FakeToken("only", "ADV"),
            FakeToken("payload", "NOUN"),
        ],
    }
    nlp = FakeNLP(docs_by_text)
    monkeypatch.setattr(spacy_service, "_load_nlp", lambda: nlp, raising=False)
    return nlp


def test_extract_syntactic_metrics_raises_or_returns(fake_nlp):
    """Verify spacy_service extracts token counts and POS distributions."""
    sample_text = "The code was written by the engineer. It runs efficiently."

    metrics = extract_syntactic_metrics(sample_text)

    assert isinstance(metrics, SyntacticMetrics)
    assert metrics.word_count == 10
    assert metrics.passive_voice_count == 1  # "was written"
    assert metrics.noun_to_verb_ratio > 0.0


def test_extract_syntactic_metrics_distinguishes_active_from_passive(fake_nlp):
    """Passive auxiliary dependency counts should not trigger for active voice."""
    passive = extract_syntactic_metrics(
        "The code was written by the engineer. It runs efficiently."
    )
    active = extract_syntactic_metrics("The engineer wrote the code. The test passes.")

    assert passive.word_count == 10
    assert passive.passive_voice_count == 1
    assert passive.noun_to_verb_ratio == pytest.approx(1.0)

    assert active.word_count == 8
    assert active.passive_voice_count == 0
    assert active.noun_to_verb_ratio == pytest.approx(1.5)


@pytest.mark.parametrize("text", ["", "   "])
def test_extract_syntactic_metrics_empty_or_whitespace_return_zeroes(text):
    """Blank inputs should not load spaCy or divide by zero."""
    metrics = extract_syntactic_metrics(text)

    assert metrics == SyntacticMetrics(
        word_count=0,
        passive_voice_count=0,
        noun_to_verb_ratio=0.0,
    )


def test_extract_syntactic_metrics_handles_non_english_tokens(fake_nlp):
    """Mixed-script payloads should tokenize without assuming English POS coverage."""
    metrics = extract_syntactic_metrics("漢字 العربية Русский text")

    assert metrics.word_count == 4
    assert metrics.passive_voice_count == 0
    assert metrics.noun_to_verb_ratio == 0.0


def test_extract_syntactic_metrics_handles_zero_verbs(fake_nlp):
    """Noun-to-verb ratio should be zero when the parsed payload has no verbs."""
    metrics = extract_syntactic_metrics("nouns only payload")

    assert metrics.word_count == 3
    assert metrics.passive_voice_count == 0
    assert metrics.noun_to_verb_ratio == 0.0
