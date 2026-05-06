"""
Called by the orchestrator before agents start.

Creates the starter repo for the textproc task:
  - document.py, tokenizer.py, utils.py  (real implementations — already exist)
  - tests/test_document.py               (existing tests — do not modify)
  - sentiment.py, keywords.py, summarizer.py, similarity.py,
    formatter.py, pipeline.py            (empty stubs — agents implement these)
"""
from pathlib import Path

# ── Existing implementations (always written; agents should NOT overwrite) ────

_EXISTING = {
    "document.py": '''\
"""document.py — Base Document class for the textproc library.

This file already has a working implementation.
Extend it by adding new properties and methods — do NOT remove or change
the existing __init__, words, or word_count.
"""


class Document:
    def __init__(self, text: str):
        self.text = text

    @property
    def words(self) -> list:
        """Whitespace-split, lowercased words (no punctuation stripping)."""
        return self.text.lower().split()

    @property
    def word_count(self) -> int:
        return len(self.words)
''',

    "tokenizer.py": '''\
"""tokenizer.py — Text tokenizer for the textproc library.

This file already has a working implementation of tokenize_words.
Extend it by adding new methods — do NOT remove or change tokenize_words.
"""
import re


class Tokenizer:
    def tokenize_words(self, text: str) -> list:
        """Lowercase, strip leading/trailing non-alphanumeric chars, split on whitespace.

        Returns cleaned lowercase tokens; empty tokens are dropped.
        """
        text = text.lower()
        tokens = text.split()
        cleaned = []
        for token in tokens:
            token = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", token)
            if token:
                cleaned.append(token)
        return cleaned
''',

    "utils.py": '''\
"""utils.py — Shared text utilities for the textproc library.

No changes needed here; import and use these in other modules as needed.
"""
import re


def clean_text(text: str) -> str:
    """Strip leading and trailing whitespace."""
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space and strip ends."""
    return re.sub(r"\\s+", " ", text).strip()
''',

    "tests/__init__.py": "",

    "tests/test_document.py": '''\
"""Tests for the base Document class (existing functionality).

These tests already pass on the starter implementation.
Do NOT modify this file.
"""
from document import Document


class TestDocument:
    def test_text_stored(self):
        doc = Document("Hello World")
        assert doc.text == "Hello World"

    def test_words_lowercased(self):
        doc = Document("Hello World")
        assert doc.words == ["hello", "world"]

    def test_word_count(self):
        doc = Document("The quick brown fox")
        assert doc.word_count == 4

    def test_empty_document_words(self):
        doc = Document("")
        assert doc.words == []

    def test_empty_document_word_count(self):
        doc = Document("")
        assert doc.word_count == 0
''',
}

# ── New stubs (written only if the file does not already exist) ───────────────

_STUBS = {
    "sentiment.py": '''\
"""sentiment.py — Sentiment analysis for the textproc library."""
from document import Document
from tokenizer import Tokenizer

POSITIVE_WORDS = {
    "good", "great", "excellent", "happy", "love", "wonderful",
    "best", "amazing", "awesome", "fantastic", "pleasant", "enjoy"
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "horrible",
    "poor", "disgusting", "unpleasant", "disappointing"
}


class SentimentAnalyzer:
    def analyze(self, doc: Document) -> dict:
        raise NotImplementedError
''',

    "keywords.py": '''\
"""keywords.py — Keyword extraction for the textproc library."""
from document import Document
from tokenizer import Tokenizer

STOPWORDS = {
    "the", "a", "an", "is", "in", "it", "of", "to", "and", "or",
    "for", "on", "at", "by", "with", "as", "be", "this", "that",
    "are", "was", "were", "i", "you", "he", "she", "we", "they",
    "not", "but", "if", "from"
}


class KeywordExtractor:
    def extract(self, doc: Document, top_k: int = 5) -> list:
        raise NotImplementedError

    def extract_ngrams(self, doc: Document, n: int, top_k: int = 5) -> list:
        raise NotImplementedError
''',

    "summarizer.py": '''\
"""summarizer.py — Extractive text summarization."""
from document import Document
from tokenizer import Tokenizer


class Summarizer:
    def summarize(self, doc: Document, n_sentences: int = 3) -> str:
        raise NotImplementedError
''',

    "similarity.py": '''\
"""similarity.py — Document similarity functions."""
from document import Document


def cosine_similarity(doc1: Document, doc2: Document) -> float:
    raise NotImplementedError


def jaccard_similarity(doc1: Document, doc2: Document) -> float:
    raise NotImplementedError
''',

    "formatter.py": '''\
"""formatter.py — Text formatting utilities."""
from utils import normalize_whitespace


class Formatter:
    def truncate(self, text: str, max_chars: int, suffix: str = "...") -> str:
        raise NotImplementedError

    def wrap(self, text: str, width: int) -> list:
        raise NotImplementedError

    def to_uppercase(self, text: str) -> str:
        raise NotImplementedError

    def to_titlecase(self, text: str) -> str:
        raise NotImplementedError

    def remove_extra_whitespace(self, text: str) -> str:
        raise NotImplementedError
''',

    "pipeline.py": '''\
"""pipeline.py — Processing pipeline for Document objects."""
from document import Document
from sentiment import SentimentAnalyzer
from keywords import KeywordExtractor
from summarizer import Summarizer


class Pipeline:
    def __init__(self, steps: list):
        self.steps = steps

    def run(self, doc: Document) -> Document:
        raise NotImplementedError


def add_sentiment(doc: Document) -> Document:
    raise NotImplementedError


def add_keywords(doc: Document, top_k: int = 5) -> Document:
    raise NotImplementedError


def add_summary(doc: Document, n_sentences: int = 3) -> Document:
    raise NotImplementedError
''',
}


def setup(repo_dir):
    repo_dir = Path(repo_dir)
    (repo_dir / "tests").mkdir(exist_ok=True)

    # Always write the existing implementations (starter code)
    for filename, content in _EXISTING.items():
        target = repo_dir / filename
        target.write_text(content)

    # Write stubs only if the file does not yet exist
    for filename, content in _STUBS.items():
        target = repo_dir / filename
        if not target.exists():
            target.write_text(content)
