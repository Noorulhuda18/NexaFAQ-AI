"""
app.py - NexaFAQ AI: Streamlit user interface.

Run with:   streamlit run app.py

Responsibilities of this file (all matching logic lives in faq_engine.py and
preprocess.py):

* page layout, sidebar and navigation (Chat / Browse FAQs / Analytics / About)
* chat history and other per-user state in ``st.session_state``
* session analytics, answer feedback, adding new FAQs
"""

from __future__ import annotations

import html
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from faq_engine import (
    DEFAULT_THRESHOLD,
    FAQ,
    FAQEngine,
    FAQError,
    FAQLoadError,
    MatchResult,
    STATUS_EMPTY,
    STATUS_LOW_CONFIDENCE,
    STATUS_NO_FAQS,
    STATUS_NO_KEYWORDS,
    STATUS_TOO_SHORT,
)
from preprocess import get_nlp_status

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "faqs.json"
CSS_PATH = BASE_DIR / "styles.css"

APP_NAME = "NexaFAQ AI"
MAX_QUESTION_CHARS = 500
NEW_CATEGORY_LABEL = "➕ New category…"
ALL_CATEGORIES_LABEL = "All categories"

PAGES = {
    "chat": ("Chat Assistant", ":material/chat_bubble:"),
    "browse": ("Browse FAQs", ":material/menu_book:"),
    "analytics": ("Analytics", ":material/bar_chart:"),
    "about": ("About Project", ":material/info:"),
}

# Inline SVG icons (Lucide style).  Drawn with currentColor so CSS controls them.
_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>'
ICONS = {
    "logo": _SVG.format('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M12 7v6M9 10h6"/>'),
    "bolt": _SVG.format('<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>'),
    "search": _SVG.format('<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>'),
    "database": _SVG.format('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>'),
    "layers": _SVG.format('<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>'),
    "info": _SVG.format('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>'),
    "alert": _SVG.format('<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>'),
}


def esc(text: object) -> str:
    """HTML-escape text and neutralise ``$`` so Streamlit does not treat prices as LaTeX."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def md_escape(text: str) -> str:
    """Escape characters that Streamlit's markdown would interpret in button/expander labels."""
    for char in ("\\", "$", "*", "_", "`", "[", "]", "~", "#"):
        text = text.replace(char, "\\" + char)
    return text


def render_html(markup: str) -> None:
    """Render an HTML snippet.

    The markup is collapsed to a single line (it must start with a block-level tag).
    That makes Markdown treat it as one raw-HTML block, so blank lines, indentation
    or characters such as ``*`` and ``_`` inside it cannot be mis-parsed.
    """
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def now_label() -> str:
    return datetime.now().strftime("%H:%M")


# --------------------------------------------------------------------------- #
# Page setup, CSS, state, engine
# --------------------------------------------------------------------------- #


def load_css() -> None:
    try:
        css = CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        st.warning("styles.css could not be loaded, so the app is shown with default styling.")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "page": "chat",
        "messages": [],  # chat history: list of dicts (see add_* helpers below)
        "log": [],  # one entry per answered question, used by the Analytics page
        "feedback": {},  # message id -> "up" / "down"
        "next_id": 1,
        "pending_question": None,  # set by buttons, consumed by the chat page
        "seed": random.randint(0, 10_000),  # keeps suggested questions stable between reruns
        "flash": None,  # (level, text) message shown after adding a FAQ
        "form_nonce": 0,  # bump to reset the "Add FAQ" form
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_engine() -> Optional[FAQEngine]:
    """Build the engine once per browser session (the TF-IDF index is not rebuilt on reruns)."""
    if "engine" in st.session_state:
        return st.session_state.engine
    try:
        engine = FAQEngine.from_file(str(FAQ_PATH))
    except FAQLoadError as exc:
        st.session_state.engine_error = str(exc)
        return None
    st.session_state.engine = engine
    st.session_state.pop("engine_error", None)
    return engine


def current_threshold() -> float:
    value = st.session_state.get("threshold", DEFAULT_THRESHOLD)
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD


# --------------------------------------------------------------------------- #
# Callbacks (run before the script reruns, so state is ready for the next render)
# --------------------------------------------------------------------------- #


def go_to(page: str) -> None:
    st.session_state.page = page


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending_question = None


def new_chat() -> None:
    reset_conversation()
    st.session_state.page = "chat"


def clear_chat() -> None:
    reset_conversation()


def ask(question: str) -> None:
    """Submit ``question`` through the normal chat pipeline (used by every suggestion button)."""
    st.session_state.pending_question = question
    st.session_state.page = "chat"


def rate(message_id: int, value: str) -> None:
    st.session_state.feedback[message_id] = value


def reset_statistics() -> None:
    st.session_state.log = []
    st.session_state.feedback = {}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def render_sidebar(engine: Optional[FAQEngine]) -> None:
    with st.sidebar:
        render_html(
            f"""<div class="brand"><div class="brand-logo">{ICONS['logo']}</div>
            <div class="brand-name">Nexa<span>FAQ</span> AI</div></div>
            <div class="brand-tag">Your intelligent knowledge assistant</div>"""
        )
        st.button("New Chat", key="new_chat", icon=":material/add:", on_click=new_chat)

        render_html('<div class="side-sep"></div><div class="side-label">Navigation</div>')
        for key, (label, icon) in PAGES.items():
            st.button(
                label,
                key=f"nav_{key}",
                icon=icon,
                type="primary" if st.session_state.page == key else "secondary",
                on_click=go_to,
                args=(key,),
            )

        render_html('<div class="side-sep"></div>')
        count = len(engine) if engine else 0
        categories = len(engine.categories) if engine else 0
        render_html(
            f"""<div class="info-card"><div class="ic-title">{ICONS['database']} Knowledge base</div>
            <div class="ic-value">{count} FAQs</div>
            <div class="ic-sub">{categories} categories · editable FAQ library</div></div>
            <div class="info-card"><div class="ic-title">{ICONS['layers']} Matching method</div>
            <div class="ic-value">TF-IDF + Cosine Similarity</div>
            <div class="ic-sub">Runs locally · no API keys</div></div>"""
        )

        st.slider(
            "Similarity threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(DEFAULT_THRESHOLD),
            step=0.05,
            format="%.2f",
            key="threshold",
            help=(
                "The minimum matching score a FAQ needs before it is returned as an answer. "
                "Higher = stricter (more fallback replies), lower = more lenient (more answers, "
                "but a higher risk of a loosely related one)."
            ),
        )
        st.button("Clear conversation", key="clear_chat", icon=":material/delete:", on_click=clear_chat)

        render_html('<div class="side-sep"></div><div class="side-footer">Powered by <b>Python &amp; NLP</b></div>')


# --------------------------------------------------------------------------- #
# Chat page
# --------------------------------------------------------------------------- #


def render_header(engine: FAQEngine) -> None:
    render_html(
        f"""<div class="hero"><h1>{APP_NAME}</h1>
        <div class="sub">Ask a question. Find the closest answer in your knowledge base.</div>
        <div class="status-pill"><span class="status-dot"></span>
        Local FAQ assistant ready · {len(engine)} FAQs loaded</div></div>"""
    )


def render_welcome(engine: FAQEngine) -> None:
    render_html(
        f"""<div class="welcome"><h2>👋 Hi, I'm your FAQ assistant</h2>
        <p>Ask a question about the topics currently covered in the knowledge base. I search the saved FAQs
        and return the closest answer when the wording is similar enough. You can expand this assistant for
        <b>education, HR, IT support, business, customer service</b> or any other field by adding your own FAQs.</p>
        <div class="demo-note">ℹ️ <span><b>Knowledge-base note.</b> This project uses local FAQ data, not a general-purpose AI model.
        It can only answer reliably when relevant questions and answers have been added to the knowledge base.</span></div></div>"""
    )
    render_html(
        f"""<div class="features">
        <div class="feature"><div class="f-icon">{ICONS['bolt']}</div><div class="f-title">Instant answers</div>
        <div class="f-text">Get a reply in a fraction of a second, no waiting in a queue.</div></div>
        <div class="feature"><div class="f-icon">{ICONS['search']}</div><div class="f-title">NLP-powered search</div>
        <div class="f-text">Your wording is cleaned, lemmatised and compared with TF-IDF and cosine similarity.</div></div>
        <div class="feature"><div class="f-icon">{ICONS['database']}</div><div class="f-title">Local knowledge base</div>
        <div class="f-text">Everything runs on this computer from a JSON file. No API keys, no cloud.</div></div>
        </div>"""
    )
    render_html('<div class="section-label">Try asking</div>')
    columns = st.columns(2)
    for index, question in enumerate(engine.suggested_questions(4, seed=st.session_state.seed)):
        with columns[index % 2]:
            st.button(md_escape(question), key=f"sugg_{index}", on_click=ask, args=(question,))


def user_bubble_html(message: Dict) -> str:
    return (
        '<div class="msg-marker msg-user"></div>'
        f'<div class="msg-text">{esc(message["content"])}</div>'
        f'<div class="msg-meta">You · {esc(message["time"])}</div>'
    )


FALLBACK_TITLES = {
    STATUS_LOW_CONFIDENCE: "No confident match found",
    STATUS_TOO_SHORT: "Could you be more specific?",
    STATUS_EMPTY: "Please type a question",
    STATUS_NO_KEYWORDS: "I need a little more to go on",
    STATUS_NO_FAQS: "The knowledge base is empty",
}


def assistant_bubble_html(message: Dict) -> str:
    if message["matched"]:
        return (
            '<div class="msg-marker msg-bot"></div>'
            f'<div class="chips"><span class="chip chip-cat">{esc(message["category"])}</span>'
            f'<span class="chip chip-score" title="How similar your wording is to the matched FAQ. '
            f'It is not the probability that the answer is correct.">'
            f'Matching score {message["score_percent"]}%</span></div>'
            f'<div class="msg-text">{esc(message["content"])}</div>'
            f'<div class="msg-meta">Matched FAQ: “{esc(message["matched_question"])}” · {esc(message["time"])}<br>'
            "The matching score shows how closely your wording fits this FAQ. "
            "It is not a probability that the answer is correct.</div>"
        )
    title = FALLBACK_TITLES.get(message["status"], "No confident match found")
    score_line = ""
    if message["status"] in (STATUS_LOW_CONFIDENCE, STATUS_TOO_SHORT):
        score_line = (
            f"Best matching score {message['score_percent']}% "
            f"(needed {message['threshold_percent']}%) · "
        )
    return (
        '<div class="msg-marker msg-bot"></div>'
        f'<div class="fallback-title">{ICONS["alert"]}{esc(title)}</div>'
        f'<div class="msg-text">{esc(message["content"])}</div>'
        f'<div class="msg-meta">{score_line}{esc(message["time"])}</div>'
    )


def render_suggestion_buttons(message: Dict, label: Optional[str]) -> None:
    if label:
        render_html(f'<div class="related-label">{esc(label)}</div>')
    for index, suggestion in enumerate(message["suggestions"]):
        st.button(
            md_escape(suggestion["question"]),
            key=f"rel_{message['id']}_{index}",
            on_click=ask,
            args=(suggestion["question"],),
        )


def render_assistant_actions(message: Dict) -> None:
    """Related questions, feedback and copy options under an assistant message."""
    if not message["matched"]:
        if message["suggestions"]:
            render_suggestion_buttons(message, "Did you mean one of these?")
        return

    feedback = st.session_state.feedback.get(message["id"])
    if feedback is None:
        left, right, _ = st.columns([1, 1, 2])
        left.button("Helpful", key=f"up_{message['id']}", icon=":material/thumb_up:", on_click=rate, args=(message["id"], "up"))
        right.button("Not helpful", key=f"down_{message['id']}", icon=":material/thumb_down:", on_click=rate, args=(message["id"], "down"))
    else:
        verdict = "helpful" if feedback == "up" else "not helpful"
        render_html(f'<div class="feedback-done">✓ Thanks! You rated this answer as {verdict}.</div>')

    with st.expander("Copy answer"):
        # st.code shows a copy button when you hover over the block - this is the
        # one clipboard method Streamlit supports natively and reliably.
        st.code(message["content"], language=None)
        st.caption("Hover over the box and click the copy icon in its top-right corner.")
    if message["suggestions"]:
        with st.expander("Related questions"):
            render_suggestion_buttons(message, None)


def render_message(message: Dict) -> None:
    if message["role"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            render_html(user_bubble_html(message))
    else:
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            render_html(assistant_bubble_html(message))
            render_assistant_actions(message)


def build_assistant_message(message_id: int, result: MatchResult) -> Dict:
    return {
        "id": message_id,
        "role": "assistant",
        "time": now_label(),
        "content": result.answer,
        "matched": result.matched,
        "status": result.status,
        "category": result.category,
        "matched_question": result.matched_question,
        "faq_id": result.faq.id if result.faq else None,
        "score_percent": result.score_percent,
        "threshold_percent": int(round(result.threshold * 100)),
        "suggestions": [{"question": s.question, "score_percent": s.score_percent} for s in result.suggestions],
    }


def process_question(engine: FAQEngine, question: str) -> None:
    """Show the user's bubble, run the matcher with a loading indicator, store everything, rerun."""
    question = question.strip()[:MAX_QUESTION_CHARS]
    user_message = {"id": st.session_state.next_id, "role": "user", "content": question, "time": now_label()}
    st.session_state.next_id += 1
    render_message(user_message)

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        render_html('<div class="msg-marker msg-bot"></div>')
        with st.spinner("Searching the knowledge base…"):
            time.sleep(0.3)  # short pause so the loading state is visible; matching itself takes milliseconds
            try:
                result = engine.match(question, threshold=current_threshold())
            except ValueError:  # invalid threshold in state: fall back to the default
                result = engine.match(question, threshold=DEFAULT_THRESHOLD)

    assistant_message = build_assistant_message(st.session_state.next_id, result)
    st.session_state.next_id += 1
    st.session_state.messages.extend([user_message, assistant_message])
    st.session_state.log.append(
        {
            "time": assistant_message["time"],
            "question": question,
            "status": result.status,
            "matched": result.matched,
            "category": result.category,
            "faq_id": assistant_message["faq_id"],
            "matched_question": result.matched_question,
            "score_percent": result.score_percent,
        }
    )
    st.rerun()


def render_chat_page(engine: FAQEngine) -> None:
    render_header(engine)

    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    typed = st.chat_input("Ask a question about the topics in the knowledge base…", max_chars=MAX_QUESTION_CHARS)
    question = pending or typed

    if not st.session_state.messages and not question:
        render_welcome(engine)
    for message in st.session_state.messages:
        render_message(message)
    if question and question.strip():
        process_question(engine, question)


# --------------------------------------------------------------------------- #
# Browse FAQs page (+ FAQ management)
# --------------------------------------------------------------------------- #


def filter_faqs(faqs: List[FAQ], query: str, category: str) -> List[FAQ]:
    """Case-insensitive search: every word of ``query`` must appear in the FAQ text."""
    words = query.lower().split()
    result = []
    for faq in faqs:
        if category != ALL_CATEGORIES_LABEL and faq.category != category:
            continue
        haystack = f"{faq.question} {faq.answer} {faq.category} {faq.keywords}".lower()
        if all(word in haystack for word in words):
            result.append(faq)
    return result


def render_browse_tab(engine: FAQEngine) -> None:
    search_col, category_col = st.columns([3, 2])
    query = search_col.text_input(
        "Search FAQs", placeholder="Search questions, answers or keywords…", key="browse_query", label_visibility="collapsed"
    )
    category = category_col.selectbox(
        "Category", [ALL_CATEGORIES_LABEL] + engine.categories, key="browse_category", label_visibility="collapsed"
    )
    matches = filter_faqs(engine.faqs, query or "", category)
    st.caption(f"Showing {len(matches)} of {len(engine)} FAQs · categories and answers come from your local knowledge base")

    if not matches:
        render_html('<div class="empty-state">No FAQs match your search. Try different words or another category.</div>')
        return

    by_category: Dict[str, List[FAQ]] = {}
    for faq in matches:
        by_category.setdefault(faq.category, []).append(faq)
    for name in sorted(by_category):
        render_html(f'<div class="section-label">{esc(name)} · {len(by_category[name])}</div>')
        for faq in by_category[name]:
            with st.expander(md_escape(faq.question)):
                render_html(
                    f'<div class="chips"><span class="chip chip-cat">{esc(faq.category)}</span></div>'
                    f'<div class="faq-answer">{esc(faq.answer)}</div>'
                )
                st.button("Ask this in chat", key=f"browse_ask_{faq.id}", icon=":material/chat_bubble:", on_click=ask, args=(faq.question,))


def render_add_tab(engine: FAQEngine) -> None:
    render_html(
        f"""<div class="notice">{ICONS['info']}<div>Add your own question and answer.
        <b>Add for this session</b> keeps it in memory only. <b>Add &amp; save to faqs.json</b> writes it to
        the file so it is still there next time. Nothing is written to disk unless you press the save button.</div></div>"""
    )
    flash = st.session_state.flash
    if flash:
        level, text = flash
        (st.success if level == "success" else st.error)(text)
        st.session_state.flash = None

    nonce = st.session_state.form_nonce
    with st.form(f"add_faq_form_{nonce}"):
        question = st.text_input("Question", max_chars=300, placeholder="e.g. Do you offer gift wrapping?")
        answer = st.text_area("Answer", max_chars=2000, height=120, placeholder="Write the answer customers should see.")
        left, right = st.columns(2)
        choice = left.selectbox("Category", engine.categories + [NEW_CATEGORY_LABEL])
        new_category = right.text_input("New category name", max_chars=40, placeholder="Only used for “New category…”")
        keywords = st.text_input(
            "Search keywords (optional)",
            max_chars=200,
            help="Extra words people may use for this topic (synonyms). They improve matching but are not shown to users.",
            placeholder="e.g. present, wrap, gift box",
        )
        session_col, save_col = st.columns(2)
        add_session = session_col.form_submit_button("Add for this session")
        add_save = save_col.form_submit_button("Add & save to faqs.json", type="primary")

    if add_session or add_save:
        category = new_category if choice == NEW_CATEGORY_LABEL else choice
        try:
            faq = engine.add_faq(question, answer, category, keywords, save_path=str(FAQ_PATH) if add_save else None)
        except FAQError as exc:  # validation, save and index errors all carry a user-friendly message
            st.error(str(exc))
        except Exception:  # never let an unexpected problem crash the page
            st.error("Something unexpected went wrong while adding the FAQ. Nothing was changed.")
        else:
            where = "and saved to faqs.json" if add_save else "for this session only"
            st.session_state.flash = ("success", f"FAQ #{faq.id} added {where}. Try asking it in the chat!")
            st.session_state.form_nonce += 1
            st.rerun()

    st.markdown("---")
    st.caption("Edited faqs.json by hand? Reload it without restarting the app.")
    if st.button("Reload faqs.json", key="reload_faqs", icon=":material/refresh:"):
        try:
            engine.reload(str(FAQ_PATH))
        except FAQError as exc:
            st.error(f"{exc} The previous FAQs are still loaded.")
        else:
            st.success(f"Reloaded {len(engine)} FAQs from faqs.json.")

    if engine.warnings:
        with st.expander(f"⚠️ {len(engine.warnings)} problem(s) found in faqs.json"):
            for warning in engine.warnings:
                st.write(f"- {warning}")


def render_browse_page(engine: FAQEngine) -> None:
    render_html(
        '<div class="page-title">Browse FAQs</div>'
        '<div class="page-sub">Search the knowledge base, filter by category or add your own questions.</div>'
    )
    browse_tab, add_tab = st.tabs(["📚 Browse", "➕ Add FAQ"])
    with browse_tab:
        render_browse_tab(engine)
    with add_tab:
        render_add_tab(engine)


# --------------------------------------------------------------------------- #
# Analytics page
# --------------------------------------------------------------------------- #


def compute_stats(log: List[Dict], feedback: Dict[int, str]) -> Dict:
    total = len(log)
    matched_entries = [entry for entry in log if entry["matched"]]
    matched = len(matched_entries)
    return {
        "total": total,
        "matched": matched,
        "fallback": total - matched,
        "match_rate": (matched / total * 100) if total else 0.0,
        "avg_score": (sum(e["score_percent"] for e in matched_entries) / matched) if matched else 0.0,
        "categories": Counter(e["category"] for e in matched_entries).most_common(6),
        "top_faqs": Counter(e["matched_question"] for e in matched_entries).most_common(5),
        "up": sum(1 for v in feedback.values() if v == "up"),
        "down": sum(1 for v in feedback.values() if v == "down"),
    }


def bars_html(rows) -> str:
    if not rows:
        return '<div class="empty-state">Nothing to show yet.</div>'
    top = max(count for _, count in rows) or 1
    return "".join(
        f'<div class="bar-row"><div class="b-label" title="{esc(name)}">{esc(name)}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{count / top * 100:.0f}%"></div></div>'
        f'<div class="b-count">{count}</div></div>'
        for name, count in rows
    )


def render_analytics_page() -> None:
    render_html(
        '<div class="page-title">Analytics</div>'
        '<div class="page-sub">How the assistant is performing in your current session.</div>'
    )
    render_html(
        f"""<div class="notice">{ICONS['info']}<div><b>Session statistics only.</b> These numbers are kept in
        memory for this browser session and disappear when you refresh or close the tab. This demo does
        <b>not</b> store persistent analytics. Tracking usage across sessions or users would need a database.</div></div>"""
    )
    stats = compute_stats(st.session_state.log, st.session_state.feedback)

    render_html(
        f"""<div class="tiles">
        <div class="tile"><div class="t-label">Questions asked</div><div class="t-value">{stats['total']}</div>
        <div class="t-sub">this session</div></div>
        <div class="tile"><div class="t-label">Matched</div><div class="t-value cyan">{stats['matched']}</div>
        <div class="t-sub">answered from the FAQs</div></div>
        <div class="tile"><div class="t-label">Fallback replies</div><div class="t-value warn">{stats['fallback']}</div>
        <div class="t-sub">no confident match</div></div>
        <div class="tile"><div class="t-label">Match rate</div><div class="t-value purple">{stats['match_rate']:.0f}%</div>
        <div class="t-sub">avg. matching score {stats['avg_score']:.0f}%</div></div></div>"""
    )

    if not stats["total"]:
        render_html('<div class="panel"><div class="empty-state">No questions yet. Ask something in the Chat Assistant and come back.</div></div>')
        return

    left, right = st.columns(2)
    with left:
        render_html(f'<div class="panel"><h3>Most frequently matched categories</h3>{bars_html(stats["categories"])}</div>')
    with right:
        render_html(f'<div class="panel"><h3>Most frequently matched FAQs</h3>{bars_html(stats["top_faqs"])}</div>')

    rated = stats["up"] + stats["down"]
    helpful = f"{stats['up'] / rated * 100:.0f}% helpful" if rated else "no ratings yet"
    render_html(
        f"""<div class="panel"><h3>Answer feedback</h3>
        <p><b>{stats['up']}</b> helpful · <b>{stats['down']}</b> not helpful · {helpful}</p></div>"""
    )

    with st.expander("Recent questions"):
        rows = "".join(
            f'<div class="recent"><span>{"✅" if entry["matched"] else "⚠️"}</span>'
            f'<div><div class="r-q">{esc(entry["question"])}</div>'
            f'<div class="msg-meta">{esc(entry["time"])} · matching score {entry["score_percent"]}%</div></div></div>'
            for entry in reversed(st.session_state.log[-10:])
        )
        render_html(f'<div class="recent-list">{rows}</div>')

    left, right = st.columns(2)
    left.download_button(
        "Download session log (JSON)",
        data=json.dumps(st.session_state.log, indent=2, ensure_ascii=False),
        file_name="nexafaq_session_log.json",
        mime="application/json",
        icon=":material/download:",
    )
    right.button("Reset session statistics", key="reset_stats", icon=":material/refresh:", on_click=reset_statistics)


# --------------------------------------------------------------------------- #
# About page
# --------------------------------------------------------------------------- #

PROJECT_TREE = """faq-chatbot/
├── app.py            Streamlit UI: navigation, chat, session state, dashboard
├── faq_engine.py     FAQ loading, TF-IDF index, cosine similarity, answers
├── preprocess.py     lowercase, tokenise, stopwords, lemmatisation
├── faqs.json         editable knowledge base (demo data)
├── styles.css        custom styling
├── requirements.txt
├── README.md
├── .streamlit/config.toml   dark theme
└── tests/test_faq_engine.py"""


def render_about_page(engine: FAQEngine) -> None:
    render_html(
        '<div class="page-title">About Project</div>'
        '<div class="page-sub">An AI-powered FAQ chatbot built with Python, NLP and Streamlit.</div>'
    )
    render_html(
        """<div class="panel"><h3>What it does</h3>
        <p><b>NexaFAQ AI</b> answers questions in natural language by finding the most similar question
        in a FAQ knowledge base. It does not use fixed if/else rules and it does not call any external AI
        service: matching is done locally with classic NLP and machine-learning tools.</p></div>"""
    )

    render_html(
        """<div class="section-label">How a question is answered</div><div class="steps">
        <div class="step"><div class="s-num">1</div><div class="s-title">Clean the text</div>
        <div class="s-text">Lowercase, remove punctuation, split into words, drop stopwords, lemmatise.</div></div>
        <div class="step"><div class="s-num">2</div><div class="s-title">Vectorise with TF-IDF</div>
        <div class="s-text">Every FAQ becomes a vector of weighted words. This is fitted once, not per message.</div></div>
        <div class="step"><div class="s-num">3</div><div class="s-title">Compare</div>
        <div class="s-text">Cosine similarity between your question and every FAQ gives a matching score.</div></div>
        <div class="step"><div class="s-num">4</div><div class="s-title">Check the threshold</div>
        <div class="s-text">Only a score at or above the sidebar threshold counts as a match.</div></div>
        <div class="step"><div class="s-num">5</div><div class="s-title">Answer or fall back</div>
        <div class="s-text">A match returns the FAQ answer. Otherwise you get a clear fallback, never a guess.</div></div>
        <div class="step"><div class="s-num">6</div><div class="s-title">Learn from feedback</div>
        <div class="s-text">Ratings and statistics for this session are shown on the Analytics page.</div></div>
        </div>"""
    )

    left, right = st.columns(2)
    with left:
        render_html(
            """<div class="panel"><h3>Technology</h3><ul>
            <li><b>Python</b> and <b>Streamlit</b> for the app and UI</li>
            <li><b>NLTK</b> for stopwords and lemmatisation</li>
            <li><b>scikit-learn</b> <code>TfidfVectorizer</code> and <code>cosine_similarity</code></li>
            <li><b>JSON</b> file as the knowledge base</li>
            <li><b>Custom CSS</b> for the interface</li></ul></div>"""
        )
    with right:
        status = get_nlp_status()
        notes = "".join(f"<li>{esc(note)}</li>" for note in status["notes"])
        health = "NLTK is fully available." if status["using_nltk"] else "Running with the built-in fallback (still fully functional)."
        render_html(
            f"""<div class="panel"><h3>NLP backend</h3>
            <p><b>Stopwords:</b> {esc(status['stopwords'])}<br><b>Lemmatizer:</b> {esc(status['lemmatizer'])}</p>
            <p>{health}</p>{f'<ul>{notes}</ul>' if notes else ''}</div>"""
        )
        if not status["using_nltk"]:
            st.caption("Install NLTK data with:  python preprocess.py --download")

    render_html(
        f"""<div class="panel"><h3>Good to know</h3><ul>
        <li><b>Knowledge base.</b> The current {len(engine)} FAQs are example content focused on a fictional online store.
        Add questions and answers from other fields in <i>Browse FAQs → Add FAQ</i> to broaden the assistant.</li>
        <li><b>Word-based matching.</b> TF-IDF compares words, not meaning. A question with no words in common
        with a FAQ can be missed. The optional <i>keywords</i> field on each FAQ helps with synonyms.</li>
        <li><b>Matching score ≠ correctness.</b> It measures how similar the wording is, not how likely the answer is right.</li>
        <li><b>Session data.</b> Chat history, statistics and feedback live only in the current browser session.</li>
        </ul></div>"""
    )
    st.markdown("**Project structure**")
    st.code(PROJECT_TREE, language=None)


# --------------------------------------------------------------------------- #
# Error page + main
# --------------------------------------------------------------------------- #


def render_load_error() -> None:
    render_html(
        f"""<div class="hero"><h1>{APP_NAME}</h1><div class="sub">Ask a question. Find the closest answer in your knowledge base.</div></div>"""
    )
    st.error(f"The FAQ knowledge base could not be loaded. {st.session_state.get('engine_error', '')}")
    render_html(
        f"""<div class="panel"><h3>How to fix it</h3><ul>
        <li>Open <code>faqs.json</code> in the project folder ({esc(FAQ_PATH)}).</li>
        <li>Make sure it is valid JSON: a list of objects, each with <code>question</code>, <code>answer</code>
        and <code>category</code>. A JSON validator or VS Code will point at the broken line.</li>
        <li>Save the file, then press <b>Try again</b>.</li></ul></div>"""
    )
    st.button("Try again", key="retry_load", icon=":material/refresh:")


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} · FAQ Assistant", page_icon="💬", layout="wide", initial_sidebar_state="expanded")
    load_css()
    init_state()

    engine = get_engine()
    render_sidebar(engine)
    if engine is None:
        render_load_error()
        return

    page = st.session_state.page
    if page == "browse":
        render_browse_page(engine)
    elif page == "analytics":
        render_analytics_page()
    elif page == "about":
        render_about_page(engine)
    else:
        render_chat_page(engine)


main()
