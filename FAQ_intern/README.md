# NexaFAQ AI — AI-Powered FAQ Chatbot (NLP + Streamlit)

NexaFAQ AI answers questions written in natural language by finding the most similar question in a FAQ knowledge base. It uses classic NLP (NLTK) and machine-learning tools (scikit-learn TF-IDF + cosine similarity). It runs **entirely on your computer**: no API keys, no cloud services, no external LLM.

The sample knowledge base is **demonstration data** for a fictional e-commerce store ("NexaShop"). The chatbot cannot look up real orders, payments or customer accounts.

---

## Features

- Natural-language questions → best matching FAQ answer, with category and **matching score**
- NLP pipeline: lowercase → clean punctuation → tokenise → stopword removal → lemmatisation
- TF-IDF index fitted **once** (not on every message) + cosine similarity
- Configurable similarity threshold (sidebar slider); low-confidence questions get a clear fallback, never a random answer
- "Did you mean…?" related questions for fallbacks, and related questions after matches
- Chat history in `st.session_state`, New Chat, Clear conversation
- Pages: **Chat Assistant**, **Browse FAQs** (search + category filter + expanders), **Analytics** (session statistics), **About Project**
- Helpful / Not helpful feedback, copy-answer option
- Add new FAQs from the UI (saved to `faqs.json` only when you press the save button)
- 52 sample FAQs in 10 categories, easy to edit
- Dark navy / purple / cyan interface with custom CSS, responsive on small screens

> **Matching score ≠ probability.** The score shown next to an answer measures how similar your wording is to the matched FAQ. It is not the probability that the answer is correct.

---

## Project structure

```
faq-chatbot/
├── app.py                  Streamlit UI: navigation, chat, session state, dashboard
├── faq_engine.py           FAQ loading, TF-IDF index, cosine similarity, answer retrieval
├── preprocess.py           NLP cleaning, tokenisation, stopwords, lemmatisation
├── faqs.json               Editable FAQ knowledge base (demo data)
├── styles.css              Custom styling
├── requirements.txt
├── README.md
├── .streamlit/config.toml  Dark theme for Streamlit's built-in widgets
└── tests/
    └── test_faq_engine.py  Automated tests (unittest, also runs under pytest)
```

---

## Setup on Windows with VS Code

### 1. Prerequisites
- **Python 3.9 or newer** from <https://www.python.org/downloads/> (tick *“Add python.exe to PATH”* during installation)
- **VS Code** with the *Python* extension

Check Python: open a terminal and run `python --version`.

### 2. Open the project
Unzip `faq-chatbot.zip`, then in VS Code choose **File → Open Folder…** and select the `faq-chatbot` folder. Open the terminal with **Terminal → New Terminal** (`` Ctrl+` ``). The terminal must be in the project folder (the one that contains `app.py`).

### 3. Create and activate a virtual environment
```bat
python -m venv .venv
.venv\Scripts\activate
```
If PowerShell refuses to run the activation script, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
(or use **Terminal → New Terminal** with *Command Prompt* instead of PowerShell).
When VS Code asks whether to use the new environment, click **Yes** (or press `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`).

### 4. Install the dependencies
```bat
pip install -r requirements.txt
```

### 5. Download the NLTK data (once, needs internet)
```bat
python preprocess.py --download
```
This downloads the NLTK `stopwords` and `wordnet` data. After that the app works fully offline. **If you skip this step the app still works**: it falls back to a built-in stopword list and a simple rule-based lemmatiser. The *About Project* page shows which backend is active.

### 6. Run the app
```bat
streamlit run app.py
```
Streamlit opens `http://localhost:8501` in your browser. Stop the server with `Ctrl+C` in the terminal.

> Run the command from the project folder so that Streamlit finds `.streamlit/config.toml` (the dark theme).

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python preprocess.py --download
streamlit run app.py
```

---

## Run the tests

```bat
python -m unittest discover -s tests -v
```
or, with pytest (installed through `requirements.txt`):
```bat
python -m pytest -v
```

The tests cover exact matching, paraphrases, irrelevant-question rejection, empty/punctuation-only/very short queries, FAQ loading and validation (missing file, empty file, malformed JSON, duplicates), threshold behaviour (including invalid values), that the vectorizer is not refitted per message, and adding / saving / reloading FAQs. They never modify your real `faqs.json`; everything that writes uses a temporary folder.

---

## Using the app

| Page | What you can do |
|---|---|
| **Chat Assistant** | Type a question and press Enter, or click a suggested question. Rate answers, open *Copy answer*, click related questions. |
| **Browse FAQs** | Search all FAQs, filter by category, expand a question to read the answer or send it to the chat. The **Add FAQ** tab lets you add your own. |
| **Analytics** | Session statistics: questions asked, matched, fallback replies, match rate, top categories and FAQs, feedback. |
| **About Project** | How the pipeline works, technology, active NLP backend, limitations. |

**Sidebar:** *New Chat* starts a fresh conversation, the **similarity threshold** slider controls how strict matching is (higher = stricter), and *Clear conversation* empties the chat. Neither deletes the FAQ dataset.

**Session vs. persistent data:** chat history, analytics and feedback live in the current browser session only and are lost when you refresh or close the tab. The only thing saved to disk is a FAQ you explicitly save to `faqs.json`.

**Copy answer:** open *Copy answer* under a reply, hover over the box and click the copy icon in its corner. Streamlit does not offer a direct clipboard API, so this built-in code-block button is the reliable method.

---

## Editing the knowledge base

`faqs.json` is a list of objects:

```json
[
  {
    "id": 1,
    "question": "How can I track my order?",
    "answer": "Sign in to your account, open My Orders and choose Track Package ...",
    "category": "Orders",
    "keywords": "where is my order package parcel shipment tracking number"
  }
]
```

- `id`, `question`, `answer` and `category` are the standard fields. Missing or duplicated ids are repaired automatically.
- `keywords` is **optional**. It lists extra words people might use (synonyms such as *parcel* for *package*). They are added to the search index but never shown to users. This is an extension to the "fit TF-IDF on the questions" approach; pass `use_keywords=False` to `FAQEngine` for pure question matching.
- Duplicate questions and invalid entries are skipped, and the problems are listed on the *Browse FAQs → Add FAQ* tab.
- After editing the file by hand, press **Reload faqs.json** there (or restart the app).

You can also add FAQs in the app: **Browse FAQs → Add FAQ**. *Add for this session* keeps the entry in memory only; *Add & save to faqs.json* writes it to the file (the file is written safely through a temporary file, and a damaged `faqs.json` is never overwritten).

---

## How the matching works

1. `preprocess.py` cleans the text and returns meaningful, lemmatised tokens. FAQ questions and user queries use exactly the same function.
2. `faq_engine.py` fits a scikit-learn `TfidfVectorizer` (word 1–2-grams) once on the preprocessed FAQs.
3. A user query is transformed with the fitted vectorizer and compared with all FAQs using `cosine_similarity`.
4. The best FAQ is returned only if its score is **at or above the threshold** (default 0.25, adjustable 0.10–0.90 in the sidebar) **and** at least 60 % of the query's meaningful words exist in the FAQ vocabulary. The second check stops a question such as *"what is your favorite color"* from matching a FAQ on the single word *color*.
5. Otherwise a fallback is shown. Single-word queries such as *"order"* are too ambiguous to answer directly, so the engine suggests related questions instead.

Edge cases handled: empty input, punctuation-only input, greetings/stopword-only input, very short and very long input, unknown questions, missing / empty / malformed FAQ file, duplicate FAQs, invalid thresholds, and save errors.

### Limitations (honest notes)
- TF-IDF compares **words**, not meaning. A paraphrase that shares no words with a FAQ can be missed — use the `keywords` field or add the paraphrase as a FAQ. A semantic model (for example Sentence Transformers) would handle this better and is a natural optional extension; it is not included so the core app stays small and fully offline.
- The default threshold was tuned on the sample FAQs. For your own data, adjust the slider.
- The Inter font is loaded from Google Fonts when you are online; offline the app falls back to your system font (Segoe UI on Windows). Everything else works offline.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick *Add python.exe to PATH*, or use `py` instead of `python`. |
| `ModuleNotFoundError: No module named 'streamlit'` (or `sklearn`, `nltk`) | The virtual environment is not active or packages are missing. Run `.venv\Scripts\activate`, then `pip install -r requirements.txt`. In VS Code select the `.venv` interpreter. |
| `'streamlit' is not recognized` | Use `python -m streamlit run app.py`. |
| PowerShell: *running scripts is disabled* | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use Command Prompt. |
| About page says *NLTK is not installed* / *data not found* | `pip install nltk`, then `python preprocess.py --download`. If the download is blocked (firewall/proxy), you can run the manual command `python -m nltk.downloader stopwords wordnet omw-1.4`. The app keeps working with the built-in fallback meanwhile. |
| `LookupError: Resource wordnet not found` in your own scripts | Same download step as above; the app itself never crashes on this. |
| Page looks plain / unstyled | Make sure `styles.css` is next to `app.py`. For the dark theme, start the app from the project folder. |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| "The FAQ knowledge base could not be loaded" | `faqs.json` is missing or not valid JSON. The message names the line; fix it and press **Try again**. |
| An old version of the app keeps showing | Press `R` in the browser tab (rerun) or stop and restart Streamlit. |

---

## Technology

Python · Streamlit · NLTK · scikit-learn (`TfidfVectorizer`, `cosine_similarity`) · NumPy · JSON · custom CSS
