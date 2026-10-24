"""
faq_engine.py - FAQ loading, TF-IDF indexing and answer retrieval.

How matching works
------------------
1. ``load_faqs`` reads and validates ``faqs.json``.
2. Every FAQ is turned into an *index document*: its question plus the optional
   ``keywords`` field (extra synonyms such as "parcel" for "package"), both run
   through the NLP pipeline in ``preprocess.py``.
3. A scikit-learn ``TfidfVectorizer`` is fitted ONCE on those documents
   (not on every user message).
4. A user question goes through the same pipeline, is transformed with the
   already-fitted vectorizer and compared to all FAQs with cosine similarity.
5. The best FAQ is returned only if its score reaches the threshold; otherwise
   the engine returns a clearly flagged fallback (never an unrelated answer).

The score is a *matching score* (how similar the wording is), not the
probability that the answer is correct.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean_text, preprocess_text, preprocess_tokens

DEFAULT_FAQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faqs.json")
DEFAULT_THRESHOLD = 0.25
# At least this share of the query's meaningful words must exist in the FAQ
# vocabulary.  Prevents "what is your favorite color" from matching on the
# single known word "color".
# A one-word query ("refund", "order") is too ambiguous to answer directly unless
# it scores at least SINGLE_WORD_MIN_SCORE; the engine offers suggestions instead.
SINGLE_WORD_MIN_SCORE = 0.5
MIN_QUERY_COVERAGE = 0.6
MAX_QUERY_CHARS = 500
SUGGESTION_MIN_SCORE = 0.10  # weakest score still shown as a "related question"
MAX_SUGGESTIONS = 3

# Limits used when validating FAQ entries.
MAX_QUESTION_CHARS = 300
MAX_ANSWER_CHARS = 2000
MAX_CATEGORY_CHARS = 40
MIN_ANSWER_CHARS = 10

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class FAQError(Exception):
    """Base class for all FAQ-related errors (messages are safe to show to users)."""


class FAQLoadError(FAQError):
    """The FAQ file is missing, empty, unreadable or malformed."""


class FAQValidationError(FAQError):
    """A single FAQ entry is invalid (empty, too long, duplicate, ...)."""


class FAQSaveError(FAQError):
    """A new FAQ could not be written to disk."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FAQ:
    id: int
    question: str
    answer: str
    category: str
    keywords: str = ""

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
        }
        if self.keywords:
            data["keywords"] = self.keywords
        return data


@dataclass(frozen=True)
class Suggestion:
    faq_id: int
    question: str
    category: str
    score: float

    @property
    def score_percent(self) -> int:
        return int(round(self.score * 100))


# Possible values of MatchResult.status
STATUS_MATCHED = "matched"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_EMPTY = "empty_query"  # nothing typed, or punctuation only
STATUS_NO_KEYWORDS = "no_keywords"  # only greetings / stopwords, e.g. "hi", "what is it"
STATUS_TOO_SHORT = "too_short"  # a single, ambiguous word such as "order"
STATUS_NO_FAQS = "no_faqs"


@dataclass
class MatchResult:
    """Outcome of ``FAQEngine.match``."""

    status: str
    message: str
    score: float = 0.0
    threshold: float = DEFAULT_THRESHOLD
    coverage: float = 0.0
    faq: Optional[FAQ] = None  # set only when status == "matched"
    suggestions: List[Suggestion] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.status == STATUS_MATCHED

    @property
    def answer(self) -> str:
        """The FAQ answer when matched, otherwise the fallback message."""
        return self.faq.answer if self.faq else self.message

    @property
    def matched_question(self) -> Optional[str]:
        return self.faq.question if self.faq else None

    @property
    def category(self) -> Optional[str]:
        return self.faq.category if self.faq else None

    @property
    def score_percent(self) -> int:
        return int(round(self.score * 100))


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def validate_threshold(value: object) -> float:
    """Return ``value`` as a float in [0, 1] or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Similarity threshold must be a number between 0 and 1, got {value!r}.")
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Similarity threshold must be between 0 and 1, got {value}.")
    return value


def question_key(question: str) -> str:
    """Case/punctuation-insensitive key used to detect duplicate questions."""
    return clean_text(question)


def validate_faq_fields(
    question: object,
    answer: object,
    category: object,
    keywords: object = "",
) -> Tuple[str, str, str, str]:
    """Validate and normalise the text fields of one FAQ.

    Returns ``(question, answer, category, keywords)`` stripped of surrounding
    whitespace, or raises ``FAQValidationError`` with a user-friendly message.
    """
    if not isinstance(question, str) or not question.strip():
        raise FAQValidationError("The question must not be empty.")
    if not isinstance(answer, str) or not answer.strip():
        raise FAQValidationError("The answer must not be empty.")
    if not isinstance(category, str) or not category.strip():
        raise FAQValidationError("The category must not be empty.")
    if keywords is None:
        keywords = ""
    if isinstance(keywords, (list, tuple)):
        keywords = " ".join(str(k) for k in keywords)
    if not isinstance(keywords, str):
        raise FAQValidationError("Keywords must be text.")

    question = re.sub(r"\s+", " ", question).strip()
    answer = answer.strip()
    category = re.sub(r"\s+", " ", category).strip()
    keywords = re.sub(r"\s+", " ", keywords).strip()

    if len(question) > MAX_QUESTION_CHARS:
        raise FAQValidationError(f"The question is too long (maximum {MAX_QUESTION_CHARS} characters).")
    if len(answer) < MIN_ANSWER_CHARS:
        raise FAQValidationError(f"The answer is too short (minimum {MIN_ANSWER_CHARS} characters).")
    if len(answer) > MAX_ANSWER_CHARS:
        raise FAQValidationError(f"The answer is too long (maximum {MAX_ANSWER_CHARS} characters).")
    if len(category) > MAX_CATEGORY_CHARS:
        raise FAQValidationError(f"The category name is too long (maximum {MAX_CATEGORY_CHARS} characters).")
    if not preprocess_tokens(question):
        raise FAQValidationError(
            "The question needs at least one meaningful word (letters or numbers, not just punctuation or filler words)."
        )
    return question, answer, category, keywords


# --------------------------------------------------------------------------- #
# Loading and saving
# --------------------------------------------------------------------------- #


def load_faqs(path: str = DEFAULT_FAQ_PATH) -> Tuple[List[FAQ], List[str]]:
    """Load and validate the FAQ file.

    Returns ``(faqs, warnings)``.  Invalid or duplicate entries are skipped and
    reported in ``warnings``.  Raises ``FAQLoadError`` if the file is missing,
    unreadable, not valid JSON, not a list, or contains no valid FAQ.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw_text = handle.read()
    except FileNotFoundError:
        raise FAQLoadError(f"FAQ file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise FAQLoadError(f"Could not read the FAQ file ({exc}).") from exc

    if not raw_text.strip():
        raise FAQLoadError("The FAQ file is empty.")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FAQLoadError(
            f"The FAQ file is not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg})."
        ) from exc
    if not isinstance(data, list):
        raise FAQLoadError("The FAQ file must contain a JSON list of FAQ objects.")

    faqs: List[FAQ] = []
    warnings: List[str] = []
    seen_questions: Dict[str, int] = {}
    used_ids: set = set()
    pending_without_id: List[Tuple[str, str, str, str]] = []

    for position, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            warnings.append(f"Entry #{position} skipped: it is not a JSON object.")
            continue
        try:
            question, answer, category, keywords = validate_faq_fields(
                item.get("question"), item.get("answer"), item.get("category"), item.get("keywords", "")
            )
        except FAQValidationError as exc:
            warnings.append(f"Entry #{position} skipped: {exc}")
            continue

        key = question_key(question)
        if key in seen_questions:
            warnings.append(f"Entry #{position} skipped: duplicate of the question '{question}'.")
            continue
        seen_questions[key] = position

        raw_id = item.get("id")
        if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id > 0 and raw_id not in used_ids:
            used_ids.add(raw_id)
            faqs.append(FAQ(raw_id, question, answer, category, keywords))
        else:
            if raw_id is not None:
                warnings.append(f"Entry #{position}: id {raw_id!r} is invalid or duplicated, a new id was assigned.")
            pending_without_id.append((question, answer, category, keywords))

    # Give entries without a usable id the next free ids (keeps file order stable).
    next_id = max(used_ids, default=0) + 1
    for question, answer, category, keywords in pending_without_id:
        faqs.append(FAQ(next_id, question, answer, category, keywords))
        next_id += 1

    if not faqs:
        raise FAQLoadError("The FAQ file does not contain any valid FAQ entries.")
    return faqs, warnings


def _read_raw_list(path: str) -> List[object]:
    """Read the FAQ file as a raw list for appending; refuse to touch a broken file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if not text.strip():
            return []
        data = json.loads(text)
    except (OSError, UnicodeDecodeError) as exc:
        raise FAQSaveError(f"Could not read the FAQ file before saving ({exc}).") from exc
    except json.JSONDecodeError as exc:
        raise FAQSaveError(
            f"The existing FAQ file is not valid JSON (line {exc.lineno}). "
            "Nothing was saved, so your file was not overwritten. Fix or restore it first."
        ) from exc
    if not isinstance(data, list):
        raise FAQSaveError("The existing FAQ file is not a JSON list. Nothing was saved.")
    return data


def save_faq_to_file(faq: FAQ, path: str = DEFAULT_FAQ_PATH) -> None:
    """Append one FAQ to the JSON file.

    The file is written to a temporary file first and then swapped in with
    ``os.replace`` so a crash can never leave a half-written FAQ file.
    """
    entries = _read_raw_list(path)
    entries.append(faq.to_dict())

    directory = os.path.dirname(os.path.abspath(path))
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".faqs-", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        raise FAQSaveError(f"Could not save the FAQ file ({exc}).") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# The matching engine
# --------------------------------------------------------------------------- #


class FAQEngine:
    """TF-IDF + cosine-similarity FAQ matcher.

    The TF-IDF vectorizer is fitted when the engine is created (and again only
    when the FAQ list changes), never per user message.
    """

    def __init__(
        self,
        faqs: Sequence[FAQ],
        threshold: float = DEFAULT_THRESHOLD,
        use_keywords: bool = True,
        min_coverage: float = MIN_QUERY_COVERAGE,
        warnings: Optional[List[str]] = None,
    ) -> None:
        self._threshold = validate_threshold(threshold)
        self.min_coverage = validate_threshold(min_coverage)
        self.use_keywords = use_keywords
        self.warnings: List[str] = list(warnings or [])
        self.faqs: List[FAQ] = list(faqs)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self.fit_count = 0  # how many times the vectorizer was fitted (see tests)
        self._build_index()

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_file(
        cls, path: str = DEFAULT_FAQ_PATH, threshold: float = DEFAULT_THRESHOLD, **kwargs
    ) -> "FAQEngine":
        faqs, warnings = load_faqs(path)
        return cls(faqs, threshold=threshold, warnings=warnings, **kwargs)

    def reload(self, path: str = DEFAULT_FAQ_PATH) -> None:
        """Re-read the FAQ file and rebuild the index.  Keeps the old data on error."""
        faqs, warnings = load_faqs(path)  # raises before anything is changed
        self.faqs = faqs
        self.warnings = warnings
        self._build_index()

    # -- properties --------------------------------------------------------- #

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = validate_threshold(value)

    def __len__(self) -> int:
        return len(self.faqs)

    @property
    def categories(self) -> List[str]:
        return sorted({faq.category for faq in self.faqs})

    # -- indexing ----------------------------------------------------------- #

    def _index_document(self, faq: FAQ) -> str:
        parts = [faq.question]
        if self.use_keywords and faq.keywords:
            parts.append(faq.keywords)
        return preprocess_text(" ".join(parts))

    def _build_index(self) -> None:
        """Preprocess all FAQs and fit the TF-IDF vectorizer (done once per FAQ set)."""
        if not self.faqs:
            self._vectorizer, self._matrix = None, None
            return
        documents = [self._index_document(faq) for faq in self.faqs]
        vectorizer = TfidfVectorizer(
            tokenizer=str.split,  # documents are already tokenised by preprocess.py
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform(documents)
        except ValueError as exc:  # e.g. "empty vocabulary"
            raise FAQLoadError(f"Could not build the search index: {exc}") from exc
        self._vectorizer, self._matrix = vectorizer, matrix
        self.fit_count += 1

    # -- matching ----------------------------------------------------------- #

    def match(self, query: object, threshold: Optional[float] = None) -> MatchResult:
        """Find the best FAQ for ``query``.

        ``threshold`` overrides the engine default for this call only (the UI
        slider uses this) and raises ``ValueError`` if it is not in [0, 1].
        """
        limit = self._threshold if threshold is None else validate_threshold(threshold)

        def result(status: str, message: str, **extra) -> MatchResult:
            return MatchResult(status=status, message=message, threshold=limit, **extra)

        if not self.faqs or self._vectorizer is None:
            return result(STATUS_NO_FAQS, "The knowledge base is empty, so I have nothing to search yet.")

        text = query[:MAX_QUERY_CHARS] if isinstance(query, str) else ""
        if not re.search(r"[A-Za-z0-9]", text):
            return result(
                STATUS_EMPTY,
                "Please type a question first. For example: “How can I track my order?”",
            )

        tokens = preprocess_tokens(text)
        if not tokens:
            return result(
                STATUS_NO_KEYWORDS,
                "That is a little too short for me to search with. Try asking a full question, "
                "such as “How do I return an item?”",
            )

        vocabulary = self._vectorizer.vocabulary_
        coverage = sum(1 for t in tokens if t in vocabulary) / len(tokens)

        query_vector = self._vectorizer.transform([" ".join(tokens)])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        order = np.argsort(-scores, kind="stable")  # stable: ties keep FAQ file order
        best = int(order[0])
        best_score = float(scores[best])

        suggestions = [
            Suggestion(self.faqs[i].id, self.faqs[i].question, self.faqs[i].category, float(scores[i]))
            for i in order[: MAX_SUGGESTIONS + 1]
            if scores[i] >= SUGGESTION_MIN_SCORE
        ]

        raw_words = re.findall(r"[A-Za-z0-9]+", text)
        if len(tokens) == 1 and len(raw_words) <= 2 and best_score < SINGLE_WORD_MIN_SCORE:
            return result(
                STATUS_TOO_SHORT,
                "Your question is very short, so I cannot tell which answer you need. "
                "Could you add a few more words? These related questions may help:",
                score=best_score,
                coverage=coverage,
                suggestions=suggestions[:MAX_SUGGESTIONS],
            )

        if best_score >= limit and coverage >= self.min_coverage:
            # offer the runners-up as "related questions", not the match itself
            related = [s for s in suggestions if s.faq_id != self.faqs[best].id][:MAX_SUGGESTIONS]
            return result(
                STATUS_MATCHED,
                "Matched a question in the knowledge base.",
                score=best_score,
                coverage=coverage,
                faq=self.faqs[best],
                suggestions=related,
            )

        return result(
            STATUS_LOW_CONFIDENCE,
            "I could not find a confident answer to that in my knowledge base, and I would rather not "
            "guess. Try rephrasing your question, or pick one of the related questions below.",
            score=best_score,
            coverage=coverage,
            suggestions=suggestions[:MAX_SUGGESTIONS],
        )

    # -- helpers for the UI ------------------------------------------------- #

    def suggested_questions(self, count: int = 4, seed: Optional[int] = None) -> List[str]:
        """Pick example questions, one per category first, so the list is varied."""
        rng = random.Random(seed)
        by_category: Dict[str, List[FAQ]] = {}
        for faq in self.faqs:
            by_category.setdefault(faq.category, []).append(faq)
        categories = list(by_category)
        rng.shuffle(categories)
        picks = [rng.choice(by_category[c]).question for c in categories[:count]]
        if len(picks) < count:  # fewer categories than requested: fill from the rest
            rest = [f.question for f in self.faqs if f.question not in picks]
            rng.shuffle(rest)
            picks.extend(rest[: count - len(picks)])
        return picks

    def add_faq(
        self,
        question: str,
        answer: str,
        category: str,
        keywords: str = "",
        save_path: Optional[str] = None,
    ) -> FAQ:
        """Add a FAQ and rebuild the index.

        If ``save_path`` is given the FAQ is also appended to that JSON file.
        The file is written first: if saving fails, ``FAQSaveError`` is raised
        and the engine is left unchanged.
        """
        question, answer, category, keywords = validate_faq_fields(question, answer, category, keywords)
        key = question_key(question)
        if any(question_key(f.question) == key for f in self.faqs):
            raise FAQValidationError("A FAQ with this question already exists.")

        new_faq = FAQ(max((f.id for f in self.faqs), default=0) + 1, question, answer, category, keywords)

        if save_path is not None:
            save_faq_to_file(new_faq, save_path)

        previous = self.faqs
        self.faqs = previous + [new_faq]
        try:
            self._build_index()
        except FAQLoadError:
            self.faqs = previous
            self._build_index()
            raise
        return new_faq
