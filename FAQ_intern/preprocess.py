"""
preprocess.py - NLP text preprocessing for NexaFAQ AI.

Pipeline (identical for FAQ questions and user queries):

    1. lowercase
    2. normalise a few phrases ("log in" -> "login", "can't" -> "can not")
    3. remove punctuation / extra whitespace
    4. tokenise
    5. remove English stopwords
    6. lemmatise (WordNet via NLTK)
    7. map a handful of closely related word forms onto one token
       (e.g. "cancellation" -> "cancel")

NLTK is the preferred backend.  If NLTK, or one of its data packages
(``stopwords`` / ``wordnet``), is missing the module does NOT crash: it
silently switches that part of the pipeline to a built-in fallback
(a bundled stopword list and a small rule-based lemmatiser).  Call
``get_nlp_status()`` to see which backend is active, and run

    python preprocess.py --download

to fetch the NLTK resources (needs internet once; afterwards everything is
offline).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, FrozenSet, List

__all__ = [
    "clean_text",
    "preprocess_tokens",
    "preprocess_text",
    "get_nlp_status",
    "download_nltk_resources",
]

# --------------------------------------------------------------------------- #
# Stopwords
# --------------------------------------------------------------------------- #

# Used only when the NLTK stopword corpus is unavailable (standard English list).
_FALLBACK_STOPWORDS: FrozenSet[str] = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves he him
    his himself she her hers herself it its itself they them their theirs themselves
    what which who whom this that these those am is are was were be been being have
    has had having do does did doing a an the and but if or because as until while of
    at by for with about against between into through during before after above below
    to from up down in out on off over under again further then once here there when
    where why how all any both each few more most other some such nor only own same
    so than too very s t can will just don should now d ll m o re ve y ain aren
    couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn
    weren won wouldn
    """.split()
)

# Conversational filler that carries no FAQ meaning.  Applied on top of
# whichever stopword list is active.
_EXTRA_STOPWORDS: FrozenSet[str] = frozenset(
    """
    please hi hello hey thanks thank want need tell know would could may might shall
    get let also ok okay
    """.split()
)

# Negations change the meaning of a question, so they are never removed.
# "where" / "when" are kept because they carry intent ("where is my order" is
# a tracking question, not an order-history question).
_KEEP_WORDS: FrozenSet[str] = frozenset({"no", "not", "nor", "where", "when"})

# --------------------------------------------------------------------------- #
# Text normalisation helpers
# --------------------------------------------------------------------------- #

# (compiled pattern, replacement) pairs applied to lower-cased text.
_PHRASE_RULES = [
    (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(r"\bcan['’]t\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"n['’]t\b"), " not"),
    (re.compile(r"\blog(?:ged|ging)?[\s-]*(?:in|into|on)\b"), "login"),
    (re.compile(r"\bsign(?:ed|ing)?[\s-]*in(?:to)?\b"), "login"),
    (re.compile(r"\bsign(?:ed|ing)?[\s-]*up\b"), "register"),
    (re.compile(r"\bsignup\b"), "register"),
    (re.compile(r"\bcheck[\s-]*out\b"), "checkout"),
    (re.compile(r"\be-mail\b"), "email"),
    (re.compile(r"\bpre-order\b"), "preorder"),
    (re.compile(r"\bt\.?f\.?a\b|\b2[\s-]?fa\b|\btwo[\s-]factor\b"), "twofactor"),
]

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_TOKEN = re.compile(r"[a-z0-9]+")

# Closely related word forms that WordNet leaves apart.  Applied AFTER
# lemmatisation, to FAQ questions and user queries alike.
_NORMALISE: Dict[str, str] = {
    "cancellation": "cancel",
    "registration": "register",
    "delivery": "deliver",
    "shipment": "ship",
    "payment": "pay",
    "refundable": "refund",
    "returnable": "return",
    "authentication": "twofactor",
    "authenticator": "twofactor",
}

_MAX_INPUT_CHARS = 2000  # guard against pathological input


def clean_text(text: object) -> str:
    """Lowercase, normalise phrases, strip punctuation and collapse whitespace.

    Non-string input (None, numbers, ...) yields an empty string.
    """
    if not isinstance(text, str):
        return ""
    text = text[:_MAX_INPUT_CHARS].lower()
    for pattern, replacement in _PHRASE_RULES:
        text = pattern.sub(replacement, text)
    text = _NON_ALNUM.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Fallback (rule-based) lemmatiser - used only without NLTK / WordNet
# --------------------------------------------------------------------------- #

_IRREGULAR: Dict[str, str] = {
    "paid": "pay", "bought": "buy", "sent": "send", "gone": "go", "went": "go",
    "children": "child", "men": "man", "women": "woman",
    "cancelled": "cancel", "cancelling": "cancel", "canceled": "cancel",
    "canceling": "cancel", "charged": "charge", "billed": "bill",
    "billing": "bill", "refunded": "refund", "shipped": "ship",
    "shipping": "ship", "forgot": "forget", "forgotten": "forget",
    "lost": "lose", "broke": "break", "broken": "break",
}


def _fallback_lemma(word: str) -> str:
    """Very small suffix-stripping lemmatiser.  Not linguistically perfect, but
    consistent: related forms of a word map to the same token."""
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) <= 3 or word.isdigit():
        return word

    base = word
    if word.endswith(("ies", "ied")) and len(word) > 4:
        base = word[:-3] + "y"
    else:
        for suffix in ("ing", "ed"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                base = word[: -len(suffix)]
                # undouble: "shipp" -> "ship", "runn" -> "run"
                if base[-1] == base[-2] and base[-1] in "bdgmnprt":
                    base = base[:-1]
                break
        else:
            if word.endswith(("sses", "xes", "zes", "ches", "shes")):
                base = word[:-2]
            elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
                base = word[:-1]

    # Drop a trailing silent "e" so "purchase"/"purchased"/"purchasing" agree.
    if base.endswith("e") and len(base) > 4:
        base = base[:-1]
    return base


# --------------------------------------------------------------------------- #
# Backend selection (NLTK if possible, otherwise fallback)
# --------------------------------------------------------------------------- #


@dataclass
class _Backend:
    stopwords: FrozenSet[str]
    lemmatize: Callable[[str], str]
    stopwords_source: str
    lemmatizer_source: str
    notes: List[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_backend() -> _Backend:
    """Detect NLTK + its data once.  Never raises."""
    notes: List[str] = []
    stopwords: FrozenSet[str] = _FALLBACK_STOPWORDS
    stopwords_source = "built-in fallback list"
    lemmatize: Callable[[str], str] = _fallback_lemma
    lemmatizer_source = "built-in rule-based fallback"

    try:
        import nltk  # noqa: F401  (imported to detect availability)
    except ImportError:
        notes.append(
            "NLTK is not installed - using the built-in fallback. "
            "Install it with: pip install nltk"
        )
        return _Backend(stopwords, lemmatize, stopwords_source, lemmatizer_source, notes)

    try:
        from nltk.corpus import stopwords as nltk_stopwords

        stopwords = frozenset(nltk_stopwords.words("english"))
        stopwords_source = "NLTK stopwords"
    except Exception:  # LookupError, OSError, ... - corpus not downloaded
        notes.append(
            "NLTK 'stopwords' data not found - using the fallback stopword list. "
            "Run: python preprocess.py --download"
        )

    try:
        from nltk.stem import WordNetLemmatizer

        _wn = WordNetLemmatizer()
        _wn.lemmatize("tests")  # forces the WordNet lookup so a missing corpus fails here

        def _nltk_lemma(word: str) -> str:
            # verb first ("shipping" -> "ship"), then noun ("orders" -> "order")
            return _wn.lemmatize(_wn.lemmatize(word, "v"), "n")

        lemmatize = _nltk_lemma
        lemmatizer_source = "NLTK WordNetLemmatizer"
    except Exception:
        notes.append(
            "NLTK 'wordnet' data not found - using the rule-based lemmatiser. "
            "Run: python preprocess.py --download"
        )

    return _Backend(stopwords, lemmatize, stopwords_source, lemmatizer_source, notes)


@lru_cache(maxsize=8192)
def _lemma(word: str) -> str:
    """Lemmatise one token (cached) and apply the small normalisation map."""
    try:
        lemma = _load_backend().lemmatize(word)
    except Exception:  # a lemmatiser glitch must never break the chatbot
        lemma = word
    return _NORMALISE.get(lemma, _NORMALISE.get(word, lemma))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def preprocess_tokens(text: object) -> List[str]:
    """Return the list of meaningful, lemmatised tokens for ``text``.

    Empty / non-string / punctuation-only input returns ``[]``.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    backend = _load_backend()
    tokens: List[str] = []
    for token in _TOKEN.findall(cleaned):
        # drop single letters (left over from "it's" etc.) but keep numbers
        if len(token) < 2 and not token.isdigit():
            continue
        is_stopword = (
            token in backend.stopwords or token in _EXTRA_STOPWORDS
        ) and token not in _KEEP_WORDS
        if is_stopword:
            continue
        lemma = _lemma(token)
        if lemma:
            tokens.append(lemma)
    return tokens


def preprocess_text(text: object) -> str:
    """Return the preprocessed text as one space-separated string."""
    return " ".join(preprocess_tokens(text))


def get_nlp_status() -> Dict[str, object]:
    """Describe which NLP backend is active (shown on the About page)."""
    backend = _load_backend()
    return {
        "stopwords": backend.stopwords_source,
        "lemmatizer": backend.lemmatizer_source,
        "using_nltk": backend.lemmatizer_source.startswith("NLTK")
        and backend.stopwords_source.startswith("NLTK"),
        "notes": list(backend.notes),
    }


def download_nltk_resources() -> bool:
    """Download the NLTK data this project uses.  Needs internet access once.

    Returns True if every resource is available afterwards.
    """
    try:
        import nltk
    except ImportError:
        print("NLTK is not installed. Run: pip install nltk")
        return False

    ok = True
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            ok = bool(nltk.download(resource, quiet=True)) and ok
        except Exception as exc:  # network / permission problems
            print(f"Could not download '{resource}': {exc}")
            ok = False
    _load_backend.cache_clear()
    _lemma.cache_clear()
    return ok


if __name__ == "__main__":
    if "--download" in sys.argv:
        print("Downloading NLTK resources..." )
        print("Done." if download_nltk_resources() else "Some resources could not be downloaded.")
    status = get_nlp_status()
    print("Stopwords :", status["stopwords"])
    print("Lemmatizer:", status["lemmatizer"])
    for note in status["notes"]:
        print(" -", note)
    sample = "How can I track my shipped orders? Can't log in!"
    print(f"\n{sample!r}\n  -> {preprocess_text(sample)!r}")
