"""
test_suite_eval.py — Hidden evaluation suite for the textproc library extension.

This suite is NOT given to agents. It tests:
  - Edge cases not covered by the public suite
  - Exact numeric values for scoring functions
  - Cross-module integration
  - Correct handling of boundary conditions

Run with: pytest test_suite_eval.py -v
"""
import math
import pytest
from collections import Counter

from document import Document
from tokenizer import Tokenizer
from sentiment import SentimentAnalyzer
from keywords import KeywordExtractor
from summarizer import Summarizer
from similarity import cosine_similarity, jaccard_similarity
from formatter import Formatter
from pipeline import Pipeline, add_sentiment, add_keywords, add_summary


# ══════════════════════════════════════════════════════════════════════════════
# TestDocumentExtensionsEval — T1, T4, T5 edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentExtensionsEval:
    def test_metadata_isolated_between_instances(self):
        doc1 = Document("hello")
        doc2 = Document("world")
        doc1.add_metadata("x", 1)
        assert "x" not in doc2.metadata

    def test_add_metadata_overwrites_existing_key(self):
        doc = Document("hello")
        doc.add_metadata("key", "first")
        doc.add_metadata("key", "second")
        assert doc.get_metadata("key") == "second"

    def test_metadata_stores_non_string_values(self):
        doc = Document("hello")
        doc.add_metadata("numbers", [1, 2, 3])
        assert doc.get_metadata("numbers") == [1, 2, 3]

    def test_char_count_multiword(self):
        # "the fox" is 7 chars including the space
        assert Document("the fox").char_count == 7

    # def test_unique_words_deduplicates(self):
    #     # UNFAIR: requires alphabetical sort on unique_words, which is undocumented
    #     doc = Document("fox fox fox bear")
    #     assert doc.unique_words == ["bear", "fox"]
    #     assert doc.unique_word_count == 2

    def test_sentence_count_matches_sentences_length(self):
        doc = Document("One. Two. Three. Four.")
        assert doc.sentence_count == len(doc.sentences)

    def test_unique_words_all_same(self):
        doc = Document("fox fox fox")
        assert doc.unique_words == ["fox"]
        assert doc.unique_word_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# TestTokenizerExtensionsEval — T2, T3 edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenizerExtensionsEval:
    def test_tokenize_sentences_question_only(self):
        t = Tokenizer()
        assert t.tokenize_sentences("Really?") == ["Really?"]

    def test_tokenize_sentences_mixed_punctuation(self):
        t = Tokenizer()
        result = t.tokenize_sentences("Wait! Are you sure? Yes.")
        assert result == ["Wait!", "Are you sure?", "Yes."]

    def test_tokenize_sentences_no_empty_strings(self):
        t = Tokenizer()
        result = t.tokenize_sentences("Hello. World.")
        assert all(len(s) > 0 for s in result)

    def test_ngrams_returns_tuples(self):
        t = Tokenizer()
        result = t.get_ngrams(["a", "b", "c"], 2)
        for item in result:
            assert isinstance(item, tuple)

    def test_ngrams_four_tokens_bigrams(self):
        t = Tokenizer()
        result = t.get_ngrams(["w", "x", "y", "z"], 2)
        assert result == [("w", "x"), ("x", "y"), ("y", "z")]

    def test_bigrams_consistent_with_ngrams(self):
        t = Tokenizer()
        tokens = ["a", "b", "c", "d"]
        assert t.get_bigrams(tokens) == t.get_ngrams(tokens, 2)

    def test_ngrams_exactly_n_tokens(self):
        # Exactly n tokens → exactly one n-gram
        t = Tokenizer()
        result = t.get_ngrams(["a", "b", "c"], 3)
        assert result == [("a", "b", "c")]

    def test_ngrams_count(self):
        # m tokens, n-gram size k → m-k+1 n-grams
        t = Tokenizer()
        tokens = list("abcdefg")  # 7 tokens
        result = t.get_ngrams(tokens, 3)
        assert len(result) == 5  # 7 - 3 + 1 = 5


# ══════════════════════════════════════════════════════════════════════════════
# TestSentimentEval — T6 edge cases and exact values
# ══════════════════════════════════════════════════════════════════════════════

class TestSentimentEval:
    def test_empty_document_neutral(self):
        doc = Document("")
        result = SentimentAnalyzer().analyze(doc)
        assert result["label"] == "neutral"
        assert result["score"] == pytest.approx(0.0)
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_case_insensitive_tokenization(self):
        # "LOVE" should be tokenized to "love" and counted as positive
        doc = Document("LOVE this")
        result = SentimentAnalyzer().analyze(doc)
        assert result["positive_count"] >= 1

    def test_result_has_all_keys(self):
        doc = Document("great")
        result = SentimentAnalyzer().analyze(doc)
        assert set(result.keys()) >= {"score", "label", "positive_count", "negative_count"}

    def test_punctuation_stripped_before_matching(self):
        # "great!" → tokenize_words gives ["great"] → 1 positive
        doc = Document("great!")
        result = SentimentAnalyzer().analyze(doc)
        assert result["positive_count"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# TestKeywordsEval — T7 edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestKeywordsEval:
    def test_top_k_zero_returns_empty(self):
        doc = Document("the dog ran fast")
        assert KeywordExtractor().extract(doc, top_k=0) == []

    def test_k_larger_than_vocab_returns_all(self):
        doc = Document("cat bear fox")
        result = KeywordExtractor().extract(doc, top_k=100)
        assert set(result) == {"cat", "bear", "fox"}

    def test_frequency_order_correct(self):
        # "dog" appears 3x, "cat" appears 1x
        doc = Document("dog dog dog cat")
        result = KeywordExtractor().extract(doc, top_k=2)
        assert result[0] == "dog"
        assert result[1] == "cat"

    def test_extract_ngrams_frequency_order(self):
        # ("fox","ran") appears once, ("the","fox") appears twice
        doc = Document("the fox the fox ran")
        result = KeywordExtractor().extract_ngrams(doc, n=2, top_k=2)
        assert result[0] == ("the", "fox")

    def test_extract_returns_list(self):
        doc = Document("cat bear fox")
        result = KeywordExtractor().extract(doc)
        assert isinstance(result, list)

    def test_extract_ngrams_empty_when_tokens_too_short(self):
        # Single word → no bigrams
        doc = Document("fox")
        result = KeywordExtractor().extract_ngrams(doc, n=2, top_k=5)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# TestSummarizerEval — T8 edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestSummarizerEval:
    def test_single_sentence_returned_as_is(self):
        doc = Document("Hello world.")
        assert Summarizer().summarize(doc, n_sentences=1) == "Hello world."

    def test_n_sentences_one_returns_best(self):
        # "the" and "fox" appear 2x → sentences 0 and 1 score highest; n=1 picks best
        doc = Document("The quick fox. The fox ran fast. A cat slept.")
        result = Summarizer().summarize(doc, n_sentences=1)
        assert isinstance(result, str)
        assert len(result) > 0

    # def test_output_is_space_joined_sentences(self):
    #     # BORDERLINE: requires sentences in original document order, not score order — undocumented
    #     doc = Document("Hello. World.")
    #     result = Summarizer().summarize(doc, n_sentences=2)
    #     assert result == "Hello. World."

    def test_n_sentences_zero_edge(self):
        doc = Document("Hello world. Foo bar.")
        result = Summarizer().summarize(doc, n_sentences=0)
        assert result == ""

    # def test_high_scoring_sentence_selected(self):
    #     # BORDERLINE: "fox" scoring 3x requires tokenizer-based (punctuation-stripped)
    #     # word counting rather than Document.words — not stated in task description
    #     # "fox" appears 3x → sentence with "fox fox fox" should score highest
    #     doc = Document("Fox fox fox. Cat ran. Dog barked.")
    #     result = Summarizer().summarize(doc, n_sentences=1)
    #     assert "Fox fox fox." in result


# ══════════════════════════════════════════════════════════════════════════════
# TestSimilarityEval — T9 edge cases and exact values
# ══════════════════════════════════════════════════════════════════════════════

class TestSimilarityEval:
    def test_cosine_empty_documents_returns_zero(self):
        doc1 = Document("")
        doc2 = Document("")
        assert cosine_similarity(doc1, doc2) == pytest.approx(0.0)

    def test_cosine_one_empty_returns_zero(self):
        doc1 = Document("fox bear")
        doc2 = Document("")
        assert cosine_similarity(doc1, doc2) == pytest.approx(0.0)

    def test_cosine_exact_value(self):
        # doc1: fox(1) bear(1), doc2: fox(1) cat(1)
        # dot = 1, |v1|=sqrt(2), |v2|=sqrt(2) → 1/2 = 0.5
        doc1 = Document("fox bear")
        doc2 = Document("fox cat")
        assert cosine_similarity(doc1, doc2) == pytest.approx(0.5)

    def test_cosine_symmetric(self):
        doc1 = Document("fox bear cat")
        doc2 = Document("bear dog fox")
        assert cosine_similarity(doc1, doc2) == pytest.approx(cosine_similarity(doc2, doc1))

    def test_jaccard_empty_documents_returns_zero(self):
        doc1 = Document("")
        doc2 = Document("")
        assert jaccard_similarity(doc1, doc2) == pytest.approx(0.0)

    def test_jaccard_exact_value(self):
        # unique_words: ["bear","fox"], ["cat","fox"]
        # intersection={"fox"}, union={"bear","cat","fox"} → 1/3
        doc1 = Document("fox bear")
        doc2 = Document("fox cat")
        assert jaccard_similarity(doc1, doc2) == pytest.approx(1 / 3)

    def test_jaccard_full_overlap(self):
        doc1 = Document("fox bear")
        doc2 = Document("bear fox fox fox")
        # unique_words of both: {"bear","fox"} → jaccard = 1.0
        assert jaccard_similarity(doc1, doc2) == pytest.approx(1.0)

    def test_cosine_repeated_words(self):
        # doc1 = "fox fox" → freq {"fox":2}
        # doc2 = "fox"     → freq {"fox":1}
        # dot=2, |v1|=2, |v2|=1 → cosine = 1.0
        doc1 = Document("fox fox")
        doc2 = Document("fox")
        assert cosine_similarity(doc1, doc2) == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TestFormatterEval — T10 edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatterEval:
    def test_truncate_exact_length_unchanged(self):
        text = "Hello"
        assert Formatter().truncate(text, 5) == "Hello"

    def test_truncate_one_over_limit(self):
        # len("Hello!")=6 > 5; text[:5-3] + "..." = text[:2] + "..." = "He..."
        assert Formatter().truncate("Hello!", 5) == "He..."

    def test_truncate_empty_suffix(self):
        # suffix="" → just truncate to max_chars
        assert Formatter().truncate("Hello World", 5, "") == "Hello"

    def test_wrap_empty_string(self):
        assert Formatter().wrap("", 10) == []

    def test_wrap_single_word_fits(self):
        assert Formatter().wrap("Hello", 10) == ["Hello"]

    # def test_wrap_single_word_exceeds_width(self):
    #     # UNFAIR: requires break_long_words=False behavior, which is undocumented
    #     # (Python's textwrap.wrap breaks long words by default)
    #     # A word longer than width goes on its own line regardless
    #     result = Formatter().wrap("Superlongword", 5)
    #     assert "Superlongword" in result

    def test_to_uppercase_preserves_spaces(self):
        assert Formatter().to_uppercase("hello world") == "HELLO WORLD"

    def test_remove_extra_whitespace_newlines(self):
        result = Formatter().remove_extra_whitespace("hello\n\nworld")
        assert result == "hello world"


# ══════════════════════════════════════════════════════════════════════════════
# TestPipelineIntegration — T11 integration across all modules
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    def test_empty_steps_returns_original(self):
        doc = Document("hello world")
        result = Pipeline([]).run(doc)
        assert isinstance(result, Document)
        assert result.text == "hello world"

    def test_all_three_steps_via_pipeline(self):
        from functools import partial
        doc = Document("The quick fox. The fox ran fast. I love great things.")
        pipeline = Pipeline([
            add_sentiment,
            partial(add_keywords, top_k=3),
            partial(add_summary, n_sentences=2),
        ])
        result = pipeline.run(doc)
        assert "sentiment" in result.metadata
        assert "keywords" in result.metadata
        assert "summary" in result.metadata

    def test_steps_run_in_order(self):
        execution_order = []

        def step_a(doc):
            execution_order.append("a")
            return doc

        def step_b(doc):
            execution_order.append("b")
            return doc

        Pipeline([step_a, step_b]).run(Document("test"))
        assert execution_order == ["a", "b"]

    def test_add_sentiment_positive_doc(self):
        doc = Document("great amazing wonderful")
        doc = add_sentiment(doc)
        assert doc.metadata["sentiment"]["label"] == "positive"

    def test_add_keywords_respects_top_k(self):
        doc = Document("cat bear fox wolf eagle")
        doc = add_keywords(doc, top_k=2)
        assert len(doc.metadata["keywords"]) == 2

    def test_add_summary_returns_non_empty_for_multi_sentence(self):
        doc = Document("The fox ran. The fox jumped. A cat slept.")
        doc = add_summary(doc, n_sentences=2)
        assert len(doc.metadata["summary"]) > 0

    def test_pipeline_metadata_accumulates_across_steps(self):
        from functools import partial
        doc = Document("I love great things the dog ran the dog")
        pipeline = Pipeline([add_sentiment, partial(add_keywords, top_k=2)])
        result = pipeline.run(doc)
        # Both keys must be present after running both steps
        assert len(result.metadata) >= 2

    def test_add_summary_metadata_is_string(self):
        doc = Document("Hello world. Foo bar baz.")
        doc = add_summary(doc, n_sentences=1)
        assert isinstance(doc.metadata["summary"], str)
