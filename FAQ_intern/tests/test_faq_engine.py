"""
Automated tests for the FAQ engine and the NLP preprocessing.

Run from the project root with either of:

    python -m unittest discover -s tests -v
    python -m pytest -v              (after: pip install pytest)

The tests are written with ``unittest`` so they run even without pytest.
They are independent of whether NLTK is installed (the engine falls back to a
built-in lemmatiser), and they never modify the real faqs.json - everything
that writes uses a temporary directory.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faq_engine import (  # noqa: E402
    DEFAULT_FAQ_PATH,
    DEFAULT_THRESHOLD,
    FAQ,
    FAQEngine,
    FAQLoadError,
    FAQSaveError,
    FAQValidationError,
    STATUS_EMPTY,
    STATUS_LOW_CONFIDENCE,
    STATUS_MATCHED,
    STATUS_NO_KEYWORDS,
    STATUS_TOO_SHORT,
    load_faqs,
    validate_threshold,
)
from preprocess import clean_text, get_nlp_status, preprocess_text, preprocess_tokens  # noqa: E402


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


class TempDirTestCase(unittest.TestCase):
    """Gives every test its own throw-away directory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="faq-tests-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def path(self, name="faqs.json"):
        return os.path.join(self.tmp, name)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


class PreprocessTests(unittest.TestCase):
    def test_lowercase_and_punctuation_removed(self):
        self.assertEqual(clean_text("  How CAN I Track,   my ORDER?!  "), "how can i track my order")

    def test_stopwords_removed(self):
        tokens = preprocess_tokens("How can I track my order?")
        self.assertIn("track", tokens)
        self.assertIn("order", tokens)
        for stopword in ("how", "can", "i", "my"):
            self.assertNotIn(stopword, tokens)

    def test_lemmatisation_unifies_word_forms(self):
        self.assertEqual(preprocess_text("orders"), preprocess_text("order"))
        self.assertEqual(preprocess_text("shipping"), preprocess_text("ship"))
        self.assertEqual(preprocess_text("returned"), preprocess_text("return"))
        self.assertEqual(preprocess_text("refunds"), preprocess_text("refund"))

    def test_phrase_normalisation(self):
        self.assertEqual(preprocess_tokens("log in"), preprocess_tokens("login"))
        self.assertEqual(preprocess_tokens("Sign in"), preprocess_tokens("login"))
        self.assertEqual(preprocess_tokens("cancellation"), preprocess_tokens("cancel"))

    def test_negation_is_kept(self):
        self.assertIn("not", preprocess_tokens("My code is not working"))
        self.assertIn("not", preprocess_tokens("my code doesn't work"))

    def test_empty_and_invalid_inputs_are_safe(self):
        for value in ("", "   ", "\n\t", None, 123, 4.5, [], {}, "?!...,,", "--- ***"):
            with self.subTest(value=value):
                self.assertEqual(preprocess_tokens(value), [])
                self.assertEqual(preprocess_text(value), "")

    def test_only_stopwords_gives_no_tokens(self):
        self.assertEqual(preprocess_tokens("what is it"), [])

    def test_same_pipeline_for_questions_and_queries(self):
        self.assertEqual(
            preprocess_text("How do I RETURN an item?"), preprocess_text("how do i return an item")
        )

    def test_status_report_has_expected_keys(self):
        status = get_nlp_status()
        for key in ("stopwords", "lemmatizer", "using_nltk", "notes"):
            self.assertIn(key, status)


# --------------------------------------------------------------------------- #
# Loading and validation of the FAQ file
# --------------------------------------------------------------------------- #


class LoadFAQTests(TempDirTestCase):
    def test_bundled_dataset_is_complete_and_valid(self):
        faqs, warnings = load_faqs(DEFAULT_FAQ_PATH)
        self.assertGreaterEqual(len(faqs), 40)
        self.assertEqual(warnings, [], "the shipped faqs.json should load without warnings")
        self.assertEqual(len({f.id for f in faqs}), len(faqs), "ids must be unique")
        self.assertEqual(len({f.question.lower() for f in faqs}), len(faqs), "questions must be unique")
        for faq in faqs:
            self.assertTrue(faq.question and faq.answer and faq.category)
        required = {
            "Orders", "Shipping", "Payments", "Returns & Refunds", "Cancellations",
            "Account", "Password", "Discounts", "Products", "Support",
        }
        self.assertTrue(required.issubset({f.category for f in faqs}))

    def test_missing_file(self):
        with self.assertRaises(FAQLoadError):
            load_faqs(self.path("does-not-exist.json"))

    def test_empty_file(self):
        write_text(self.path(), "   \n")
        with self.assertRaisesRegex(FAQLoadError, "empty"):
            load_faqs(self.path())

    def test_malformed_json(self):
        write_text(self.path(), '[{"id": 1, "question": "Oops",')
        with self.assertRaisesRegex(FAQLoadError, "not valid JSON"):
            load_faqs(self.path())

    def test_json_must_be_a_list(self):
        write_json(self.path(), {"question": "x"})
        with self.assertRaisesRegex(FAQLoadError, "list"):
            load_faqs(self.path())

    def test_empty_list_and_all_invalid_entries(self):
        write_json(self.path(), [])
        with self.assertRaises(FAQLoadError):
            load_faqs(self.path())
        write_json(self.path(), [{"question": ""}, "text", 5, {"answer": "x" * 20}])
        with self.assertRaisesRegex(FAQLoadError, "valid FAQ"):
            load_faqs(self.path())

    def test_invalid_entries_are_skipped_with_warnings(self):
        write_json(
            self.path(),
            [
                {"id": 1, "question": "How do I pay?", "answer": "Use a card at checkout.", "category": "Payments"},
                {"id": 2, "question": "", "answer": "Missing question text here.", "category": "Payments"},
                {"id": 3, "question": "No answer?", "answer": "", "category": "Payments"},
                {"id": 4, "question": "No category?", "answer": "Some valid answer text.", "category": ""},
                "not an object",
            ],
        )
        faqs, warnings = load_faqs(self.path())
        self.assertEqual([f.id for f in faqs], [1])
        self.assertEqual(len(warnings), 4)

    def test_duplicate_questions_are_skipped(self):
        write_json(
            self.path(),
            [
                {"id": 1, "question": "How do I pay?", "answer": "Use a card at checkout.", "category": "Payments"},
                {"id": 2, "question": "how do i PAY", "answer": "A different answer text.", "category": "Payments"},
            ],
        )
        faqs, warnings = load_faqs(self.path())
        self.assertEqual(len(faqs), 1)
        self.assertTrue(any("duplicate" in w.lower() for w in warnings))

    def test_missing_and_duplicate_ids_are_repaired(self):
        write_json(
            self.path(),
            [
                {"id": 5, "question": "First question here?", "answer": "First answer text.", "category": "A"},
                {"id": 5, "question": "Second question here?", "answer": "Second answer text.", "category": "A"},
                {"question": "Third question here?", "answer": "Third answer text.", "category": "A"},
            ],
        )
        faqs, _ = load_faqs(self.path())
        ids = [f.id for f in faqs]
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids[0], 5)

    def test_engine_from_file_propagates_load_errors(self):
        with self.assertRaises(FAQLoadError):
            FAQEngine.from_file(self.path("nope.json"))


# --------------------------------------------------------------------------- #
# Matching behaviour on the bundled dataset
# --------------------------------------------------------------------------- #


class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FAQEngine.from_file(DEFAULT_FAQ_PATH)

    def test_exact_questions_match_themselves(self):
        for faq in self.engine.faqs:
            with self.subTest(question=faq.question):
                result = self.engine.match(faq.question)
                self.assertTrue(result.matched)
                self.assertEqual(result.faq.id, faq.id)
                self.assertEqual(result.answer, faq.answer)
                self.assertEqual(result.matched_question, faq.question)
                self.assertEqual(result.category, faq.category)

    def test_exact_match_ignores_case_and_punctuation(self):
        result = self.engine.match("HOW CAN I TRACK MY ORDER")
        self.assertTrue(result.matched)
        self.assertEqual(result.faq.question, "How can I track my order?")

    def test_paraphrased_questions_match_the_right_faq(self):
        by_question = {f.question: f.id for f in self.engine.faqs}
        cases = {
            "I want to track my parcel": "How can I track my order?",
            "how do i reset a forgotten password": "I forgot my password. How do I reset it?",
            "Is there a discount for students?": "Do you offer student or military discounts?",
            "how to register an account": "How do I create an account?",
            "cancel my order please": "How do I cancel my order?",
            "how to sign in": "How do I log in to my account?",
            "what is your policy for returns": "What is your return policy?",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                result = self.engine.match(query)
                self.assertTrue(result.matched, f"{query!r} was not matched (score {result.score:.2f})")
                self.assertEqual(result.faq.id, by_question[expected])

    def test_irrelevant_questions_are_rejected(self):
        irrelevant = [
            "What is the capital of France?",
            "Who won the football world cup?",
            "How do I bake a chocolate cake?",
            "what is the meaning of life",
            "How tall is Mount Everest?",
            "what is your favorite color",
            "tell me a joke",
            "how to fix my car engine",
        ]
        for query in irrelevant:
            with self.subTest(query=query):
                result = self.engine.match(query)
                self.assertFalse(result.matched)
                self.assertIsNone(result.faq, "no FAQ may be attached to a rejected question")
                self.assertEqual(result.status, STATUS_LOW_CONFIDENCE)
                self.assertEqual(result.answer, result.message)  # fallback text, not a FAQ answer

    def test_low_confidence_results_offer_related_questions_only_when_relevant(self):
        result = self.engine.match("What is the capital of France?")
        self.assertEqual(result.suggestions, [])
        result = self.engine.match("my order payment")  # relevant words, vague question
        for suggestion in result.suggestions:
            self.assertGreaterEqual(suggestion.score, 0.10)

    def test_matched_result_lists_other_related_questions(self):
        result = self.engine.match("How do I return an item?")
        self.assertTrue(result.matched)
        self.assertNotIn(result.faq.id, [s.faq_id for s in result.suggestions])
        self.assertLessEqual(len(result.suggestions), 3)

    def test_score_is_a_valid_percentage(self):
        result = self.engine.match("How can I track my order?")
        self.assertGreater(result.score, 0.5)
        self.assertLessEqual(result.score, 1.0)
        self.assertEqual(result.score_percent, int(round(result.score * 100)))

    def test_empty_and_punctuation_only_queries(self):
        for query in ("", "   ", "\n", None, 42, "?", "?!?!", "...", "---", "@#$%"):
            with self.subTest(query=query):
                result = self.engine.match(query)
                self.assertEqual(result.status, STATUS_EMPTY)
                self.assertFalse(result.matched)
                self.assertTrue(result.message)

    def test_greetings_and_filler_have_no_keywords(self):
        for query in ("hi", "hello there", "a", "what is it", "please"):
            with self.subTest(query=query):
                self.assertEqual(self.engine.match(query).status, STATUS_NO_KEYWORDS)

    def test_very_short_ambiguous_queries(self):
        for query in ("order", "refund", "payment"):
            with self.subTest(query=query):
                result = self.engine.match(query)
                self.assertEqual(result.status, STATUS_TOO_SHORT)
                self.assertFalse(result.matched)
                self.assertTrue(result.suggestions, "should offer related questions instead")

    def test_two_word_query_can_match(self):
        self.assertTrue(self.engine.match("return policy").matched)

    def test_very_long_query_does_not_crash(self):
        result = self.engine.match("how do I track my order " * 500)
        self.assertIn(result.status, {STATUS_MATCHED, STATUS_LOW_CONFIDENCE})

    def test_unicode_and_odd_input_does_not_crash(self):
        for query in ("¿Cómo rastreo mi pedido?", "追踪我的订单", "💥💥💥", "<script>alert(1)</script>", "order\x00 status"):
            with self.subTest(query=query):
                result = self.engine.match(query)
                self.assertIsNotNone(result.status)


# --------------------------------------------------------------------------- #
# Threshold behaviour
# --------------------------------------------------------------------------- #


class ThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FAQEngine.from_file(DEFAULT_FAQ_PATH)

    def test_default_threshold(self):
        self.assertEqual(self.engine.threshold, DEFAULT_THRESHOLD)

    def test_maximum_threshold_rejects_a_paraphrase(self):
        paraphrase = "I want to track my parcel"
        self.assertTrue(self.engine.match(paraphrase).matched)
        strict = self.engine.match(paraphrase, threshold=1.0)
        self.assertFalse(strict.matched)
        self.assertEqual(strict.status, STATUS_LOW_CONFIDENCE)
        self.assertEqual(strict.threshold, 1.0)

    def test_a_threshold_equal_to_the_score_accepts(self):
        score = self.engine.match("I want to track my parcel").score
        self.assertTrue(self.engine.match("I want to track my parcel", threshold=score).matched)
        self.assertFalse(self.engine.match("I want to track my parcel", threshold=min(1.0, score + 0.01)).matched)

    def test_zero_threshold_still_needs_overlap(self):
        # Even with threshold 0 an unrelated question must not get an answer:
        # the coverage check rejects queries made of unknown words.
        self.assertFalse(self.engine.match("What is the capital of France?", threshold=0.0).matched)

    def test_per_call_threshold_does_not_change_engine_default(self):
        self.engine.match("track my order", threshold=0.9)
        self.assertEqual(self.engine.threshold, DEFAULT_THRESHOLD)

    def test_threshold_setter(self):
        engine = FAQEngine.from_file(DEFAULT_FAQ_PATH, threshold=0.2)
        engine.threshold = 0.6
        self.assertEqual(engine.threshold, 0.6)
        self.assertEqual(engine.match("track my order").threshold, 0.6)

    def test_invalid_thresholds_are_rejected(self):
        bad_values = [-0.1, 1.01, 5, float("nan"), float("inf"), "0.5", None, True, [], object()]
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_threshold(value)
        for value in (-1, 2, float("nan"), "abc"):
            with self.subTest(constructor=value):
                with self.assertRaises(ValueError):
                    FAQEngine.from_file(DEFAULT_FAQ_PATH, threshold=value)
            with self.subTest(match=value):
                with self.assertRaises(ValueError):
                    self.engine.match("track my order", threshold=value)
        with self.assertRaises(ValueError):
            self.engine.threshold = 3

    def test_boundary_thresholds_are_valid(self):
        self.assertEqual(validate_threshold(0), 0.0)
        self.assertEqual(validate_threshold(1), 1.0)
        self.assertTrue(math.isclose(validate_threshold(0.35), 0.35))


# --------------------------------------------------------------------------- #
# Index behaviour, adding and reloading FAQs
# --------------------------------------------------------------------------- #


def small_faqs():
    return [
        FAQ(1, "How do I pay for my order?", "Use a card at checkout.", "Payments"),
        FAQ(2, "How do I return a product?", "Start a return from your account.", "Returns"),
    ]


class IndexAndManagementTests(TempDirTestCase):
    def make_file(self, entries=None):
        entries = entries if entries is not None else [f.to_dict() for f in small_faqs()]
        write_json(self.path(), entries)
        return self.path()

    def test_vectorizer_is_fitted_once_not_per_message(self):
        engine = FAQEngine.from_file(DEFAULT_FAQ_PATH)
        self.assertEqual(engine.fit_count, 1)
        for query in ("track my order", "forgot my password", "pizza", "return policy", ""):
            engine.match(query)
        self.assertEqual(engine.fit_count, 1)

    def test_empty_faq_list_gives_clear_status(self):
        engine = FAQEngine([])
        self.assertEqual(engine.match("anything").status, "no_faqs")

    def test_add_faq_in_memory_is_immediately_searchable(self):
        engine = FAQEngine(small_faqs())
        self.assertFalse(engine.match("Do you gift wrap presents?").matched)
        added = engine.add_faq("Do you offer gift wrapping?", "Yes, for $3 per item at checkout.", "Orders")
        self.assertEqual(added.id, 3)
        self.assertEqual(len(engine), 3)
        self.assertEqual(engine.fit_count, 2)  # refitted once because the FAQ set changed
        result = engine.match("gift wrapping")
        self.assertTrue(result.matched)
        self.assertEqual(result.faq.id, 3)

    def test_add_faq_rejects_duplicates_and_invalid_fields(self):
        engine = FAQEngine(small_faqs())
        with self.assertRaisesRegex(FAQValidationError, "already exists"):
            engine.add_faq("how do i PAY for my order", "Another answer text here.", "Payments")
        invalid = [
            ("", "Valid answer text.", "Cat"),
            ("   ", "Valid answer text.", "Cat"),
            ("Valid question?", "", "Cat"),
            ("Valid question?", "short", "Cat"),
            ("Valid question?", "Valid answer text.", ""),
            ("?!?!", "Valid answer text.", "Cat"),
            ("the of and", "Valid answer text.", "Cat"),
            ("q" * 301, "Valid answer text.", "Cat"),
            ("Valid question?", "a" * 2001, "Cat"),
            ("Valid question?", "Valid answer text.", "c" * 41),
            (None, "Valid answer text.", "Cat"),
        ]
        for question, answer, category in invalid:
            with self.subTest(question=question, answer=answer, category=category):
                with self.assertRaises(FAQValidationError):
                    engine.add_faq(question, answer, category)
        self.assertEqual(len(engine), 2, "failed additions must not change the engine")

    def test_ids_continue_after_the_highest_existing_id(self):
        engine = FAQEngine([FAQ(7, "First question here?", "First answer text.", "A")])
        self.assertEqual(engine.add_faq("Second question here?", "Second answer text.", "A").id, 8)

    def test_add_faq_with_save_persists_and_reloads(self):
        path = self.make_file()
        engine = FAQEngine.from_file(path)
        engine.add_faq(
            "Do you offer gift wrapping?", "Yes, for $3 per item at checkout.", "Orders",
            keywords="present wrap", save_path=path,
        )
        reloaded = FAQEngine.from_file(path)
        self.assertEqual(len(reloaded), 3)
        self.assertEqual(reloaded.faqs[-1].keywords, "present wrap")
        self.assertTrue(reloaded.match("gift wrapping").matched)
        # the original entries are untouched
        self.assertEqual([f.id for f in reloaded.faqs], [1, 2, 3])
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(len(json.load(handle)), 3)

    def test_nothing_is_written_without_save_path(self):
        path = self.make_file()
        engine = FAQEngine.from_file(path)
        engine.add_faq("Do you offer gift wrapping?", "Yes, for $3 per item at checkout.", "Orders")
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(len(json.load(handle)), 2)

    def test_saving_creates_a_missing_file(self):
        engine = FAQEngine(small_faqs())
        new_path = self.path("new_faqs.json")
        engine.add_faq("Do you offer gift wrapping?", "Yes, for $3 per item at checkout.", "Orders", save_path=new_path)
        self.assertTrue(os.path.exists(new_path))
        with open(new_path, encoding="utf-8") as handle:
            self.assertEqual(len(json.load(handle)), 1)

    def test_save_refuses_to_overwrite_a_malformed_file(self):
        path = self.path()
        broken = '[{"id": 1, "question": "Broken'
        write_text(path, broken)
        engine = FAQEngine(small_faqs())
        with self.assertRaises(FAQSaveError):
            engine.add_faq("Do you offer gift wrapping?", "Yes, for $3 per item at checkout.", "Orders", save_path=path)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), broken, "a damaged file must be left exactly as it was")
        self.assertEqual(len(engine), 2, "engine must be unchanged when saving fails")

    def test_save_refuses_a_file_that_is_not_a_list(self):
        path = self.path()
        write_json(path, {"oops": True})
        engine = FAQEngine(small_faqs())
        with self.assertRaises(FAQSaveError):
            engine.add_faq("Do you offer gift wrapping?", "Yes, for $3 per item at checkout.", "Orders", save_path=path)

    def test_save_to_unwritable_location_raises_save_error(self):
        engine = FAQEngine(small_faqs())
        bad_path = os.path.join(self.tmp, "no_such_folder", "faqs.json")
        with self.assertRaises(FAQSaveError):
            engine.add_faq("Do you offer gift wrapping?", "Yes, for $3 per item at checkout.", "Orders", save_path=bad_path)
        self.assertEqual(len(engine), 2)

    def test_no_temp_files_are_left_behind(self):
        path = self.make_file()
        engine = FAQEngine.from_file(path)
        engine.add_faq("Do you offer gift wrapping?", "Yes, for $3 per item at checkout.", "Orders", save_path=path)
        self.assertEqual(os.listdir(self.tmp), ["faqs.json"])

    def test_reload_picks_up_external_edits(self):
        path = self.make_file()
        engine = FAQEngine.from_file(path)
        self.assertEqual(len(engine), 2)
        entries = [f.to_dict() for f in small_faqs()]
        entries.append(
            {"id": 3, "question": "Do you sell gift cards?", "answer": "Yes, from $10 to $200.", "category": "Products"}
        )
        write_json(path, entries)
        engine.reload(path)
        self.assertEqual(len(engine), 3)
        self.assertTrue(engine.match("do you sell gift cards").matched)

    def test_failed_reload_keeps_the_previous_faqs(self):
        path = self.make_file()
        engine = FAQEngine.from_file(path)
        write_text(path, "{ definitely not json")
        with self.assertRaises(FAQLoadError):
            engine.reload(path)
        self.assertEqual(len(engine), 2)
        self.assertTrue(engine.match("How do I return a product?").matched)

    def test_categories_and_suggested_questions(self):
        engine = FAQEngine.from_file(DEFAULT_FAQ_PATH)
        self.assertIn("Orders", engine.categories)
        suggestions = engine.suggested_questions(4, seed=1)
        self.assertEqual(len(suggestions), 4)
        self.assertEqual(len(set(suggestions)), 4)
        known = {f.question for f in engine.faqs}
        self.assertTrue(all(q in known for q in suggestions))
        self.assertEqual(suggestions, engine.suggested_questions(4, seed=1), "same seed, same suggestions")
        # works even with fewer FAQs than requested
        self.assertEqual(len(FAQEngine(small_faqs()).suggested_questions(4, seed=1)), 2)

    def test_keywords_improve_synonym_matching(self):
        faqs = [
            FAQ(1, "How can I track my order?", "Use the Orders page.", "Orders", "parcel package whereabouts"),
            FAQ(2, "How do I pay for my order?", "Use a card at checkout.", "Payments"),
        ]
        with_kw = FAQEngine(faqs, use_keywords=True).match("parcel whereabouts")
        self.assertTrue(with_kw.matched)
        self.assertEqual(with_kw.faq.id, 1)
        without_kw = FAQEngine(faqs, use_keywords=False).match("parcel whereabouts")
        self.assertFalse(without_kw.matched)


if __name__ == "__main__":
    unittest.main(verbosity=2)
