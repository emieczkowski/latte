"""
test_suite.py — Public test suite for the textproc library extension.

Covers all 11 implementation tasks:
  T1  Document metadata           (document.py)
  T2  Tokenizer sentence split    (tokenizer.py)
  T3  Tokenizer n-grams           (tokenizer.py)
  T4  Document sentence fields    (document.py)
  T5  Document text statistics    (document.py)
  T6  SentimentAnalyzer           (sentiment.py)
  T7  KeywordExtractor            (keywords.py)
  T8  Summarizer                  (summarizer.py)
  T9  cosine/jaccard similarity   (similarity.py)
  T10 Formatter                   (formatter.py)
  T11 Pipeline                    (pipeline.py)

Run with: pytest test_suite.py -v
"""
import pytest

from document import Document
from tokenizer import Tokenizer
from sentiment import SentimentAnalyzer
from keywords import KeywordExtractor
from summarizer import Summarizer
from similarity import cosine_similarity, jaccard_similarity
from formatter import Formatter
from pipeline import Pipeline, add_sentiment, add_keywords, add_summary


def test_required_interfaces_exist():
    # Document
    doc = Document("test")
    assert hasattr(doc, "add_metadata")
    assert hasattr(doc, "get_metadata")
    assert hasattr(doc, "sentences")
    assert hasattr(doc, "sentence_count")
    assert hasattr(doc, "char_count")
    assert hasattr(doc, "unique_words")
    assert hasattr(doc, "unique_word_count")

    # Tokenizer
    t = Tokenizer()
    assert hasattr(t, "tokenize_sentences")
    assert hasattr(t, "get_ngrams")
    assert hasattr(t, "get_bigrams")

    # Sentiment
    s = SentimentAnalyzer()
    assert hasattr(s, "analyze")

    # Keywords
    k = KeywordExtractor()
    assert hasattr(k, "extract")
    assert hasattr(k, "extract_ngrams")

    # Summarizer
    sm = Summarizer()
    assert hasattr(sm, "summarize")

    # Similarity
    assert callable(cosine_similarity)
    assert callable(jaccard_similarity)

    # Formatter
    f = Formatter()
    assert hasattr(f, "truncate")
    assert hasattr(f, "wrap")
    assert hasattr(f, "to_uppercase")
    assert hasattr(f, "to_titlecase")
    assert hasattr(f, "remove_extra_whitespace")

    # Pipeline
    p = Pipeline([])
    assert hasattr(p, "run")
    assert callable(add_sentiment)
    assert callable(add_keywords)
    assert callable(add_summary)