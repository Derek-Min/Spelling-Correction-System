import re
from urllib.parse import quote, unquote

import spacy
import streamlit as st
from collections import Counter

# ----------------------------------------------------
# Streamlit Config
# ----------------------------------------------------
st.set_page_config(page_title="Spelling Correction System", layout="wide")

# ----------------------------------------------------
# Handle replacement from query params
# ----------------------------------------------------
if "text" not in st.session_state:
    st.session_state.text = ""

def replace_word(idx: int, new_word: str):
    words = st.session_state.text.split()
    if 0 <= idx < len(words):
        words[idx] = new_word
        st.session_state.text = " ".join(words)

# Check query params
query_params = st.query_params
if "replace_idx" in query_params and "replace_word" in query_params:
    try:
        idx = int(query_params["replace_idx"])
        new_word = unquote(query_params["replace_word"])
        replace_word(idx, new_word)
        # Clear query params
        st.query_params.clear()
        # Force recheck
        if st.session_state.text:
            st.session_state.should_recheck = True
        st.rerun()
    except Exception as e:
        pass

# ----------------------------------------------------
# CSS Styling
# ----------------------------------------------------
st.markdown(
    """
<style>
:root {
    --bg: #2B2B2B;
    --panel: #313335;
    --card: #3C3F41;
    --text: #A9B7C6;
    --border: #4B4B4B;
    --err: #FF6B68;
    --real: #FFC66D;
    --grammar: #4EC9B0;
}

body, .main {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Segoe UI', sans-serif;
}

.box {
    background-color: var(--card);
    border: 1px solid var(--border);
    padding: 12px;
    border-radius: 8px;
    margin-top: 8px;
}

span.err {
    position: relative;
    padding-bottom: 2px;
    background-size: 6px 6px;
    background-repeat: repeat-x;
    background-position-y: bottom;
    cursor: pointer;
}

span.nonword {
    background-image: repeating-linear-gradient(
        -45deg,
        transparent 0 3px,
        var(--err) 3px 6px
    );
}

span.realword {
    background-image: repeating-linear-gradient(
        -45deg,
        transparent 0 3px,
        var(--real) 3px 6px
    );
}

span.grammar {
    background-image: repeating-linear-gradient(
        -45deg,
        transparent 0 3px,
        var(--grammar) 3px 6px
    );
}

span.err .popup {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    top: 1.4em;
    left: 0;
    background: #111;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 4px;
    z-index: 9999;
    white-space: nowrap;
    transition: opacity 0.15s ease-in;
}

span.err:hover .popup {
    visibility: visible;
    opacity: 1;
}

.poplink {
    cursor: pointer;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    display: block;
    text-decoration: none;
    margin: 2px 0;
}

.poplink:hover {
    background: #444;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# Load NLP + Business Corpus
# ----------------------------------------------------
@st.cache_resource
def load_resources():
    nlp = spacy.load("en_core_web_sm")
    with open("clean_business_corpus.txt", "r", encoding="utf8") as f:
        corpus = f.read()
    words = re.findall(r"[a-z]+", corpus.lower())
    word_freq = Counter(words)
    dict_words = sorted(word_freq.keys())
    return nlp, word_freq, dict_words

nlp, word_freq, dictionary_words = load_resources()

# ----------------------------------------------------
# Confusion sets
# ----------------------------------------------------
CONFUSION_SETS = {
    "sea": ["see"], "see": ["sea"], "form": ["from"], "from": ["form"],
    "their": ["there"], "there": ["their"], "to": ["too"], "too": ["to"],
}

# ----------------------------------------------------
# Edit Distance
# ----------------------------------------------------
def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]

# ----------------------------------------------------
# Ranked Suggestions
# ----------------------------------------------------
def ranked_suggestions(word, max_suggestions=3):
    candidates = []
    for w in word_freq.keys():
        if abs(len(w) - len(word)) <= 2:
            d = edit_distance(word, w)
            if d <= 2:
                candidates.append((w, d, -word_freq[w]))
    candidates.sort(key=lambda x: (x[1], x[2]))
    return [c[0] for c in candidates[:max_suggestions]]

# ----------------------------------------------------
# Check text
# ----------------------------------------------------
def check_text(text: str):
    doc = nlp(text)
    tokens = [t for t in doc if t.is_alpha]
    errors = {}

    for tok in tokens:
        lw = tok.text.lower()
        if lw not in word_freq:
            errors[lw] = {"kind": "non-word", "sugs": ranked_suggestions(lw) or ["(no suggestion)"]}

    for tok in tokens:
        lw = tok.text.lower()
        if lw not in errors and lw in CONFUSION_SETS:
            errors[lw] = {"kind": "real-word", "sugs": CONFUSION_SETS[lw]}

    singular = {"he", "she", "it", "this", "that"}
    plural = {"i", "we", "they", "you", "these", "those"}

    for tok in doc:
        if tok.tag_ in ("VBP", "VBZ"):
            subj = None
            for child in tok.children:
                if child.dep_ == "nsubj":
                    subj = child
                    break
            if not subj:
                continue
            subj_l = subj.text.lower()
            verb_l = tok.text.lower()
            if verb_l in errors:
                continue
            sugs = []
            if subj_l in singular and tok.tag_ == "VBP":
                sugs = [tok.lemma_ + "s"]
            elif subj_l in plural and tok.tag_ == "VBZ":
                sugs = [tok.lemma_]
            if sugs:
                errors[verb_l] = {"kind": "grammar", "sugs": sugs}

    return errors

# ----------------------------------------------------
# Highlight with clickable links
# ----------------------------------------------------
def highlight(text: str, errors: dict) -> str:
    words = text.split()
    html_words = []

    for idx, w in enumerate(words):
        clean = re.sub(r"[^\w']+", "", w.lower())

        if clean in errors:
            info = errors[clean]
            kind = info["kind"]
            sugs = info["sugs"]

            if not sugs:
                html_words.append(w)
                continue

            if kind == "non-word":
                css = "nonword"
            elif kind == "real-word":
                css = "realword"
            else:
                css = "grammar"

            m = re.match(r"([A-Za-z']+)([^A-Za-z']*)$", w)
            if m:
                trail = m.group(2)
            else:
                trail = ""

            popup_items = []
            for s in sugs:
                repl = s + trail
                # Use onclick with JavaScript to update URL and reload
                onclick = f"event.preventDefault(); window.location.href = '?replace_idx={idx}&replace_word={quote(repl)}';"
                popup_items.append(f"<a href='#' onclick=\"{onclick}\" class='poplink'>{repl}</a>")

            popup_html = "<span class='popup'>" + "".join(popup_items) + "</span>"
            html_words.append(f"<span class='err {css}'>{w}{popup_html}</span>")
        else:
            html_words.append(w)

    return "<div class='box'>" + " ".join(html_words) + "</div>"

# ----------------------------------------------------
# Apply corrections
# ----------------------------------------------------
def apply_corrections(text: str, errors: dict) -> str:
    return text

# ----------------------------------------------------
# UI LAYOUT
# ----------------------------------------------------
left, right = st.columns([3, 1.2])

with left:
    st.markdown("## Spelling & Grammar Checker")

    col1, col2 = st.columns(2)
    with col1:
        run = st.button("Check Text")
    with col2:
        clear = st.button("Clear")

    if clear:
        st.session_state.text = ""
        st.session_state.errors = {}
        st.rerun()

    text_input = st.text_area("Main Text", st.session_state.text, height=200)
    st.session_state.text = text_input

    if run or st.session_state.get("should_recheck", False):
        st.session_state.errors = check_text(text_input)
        st.session_state.should_recheck = False

    errors = st.session_state.get("errors", {})

    # ---------------- Highlighted Text ----------------
    st.markdown("### Highlighted Text")
    if errors:
        html = highlight(text_input, errors)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("<div class='box'>No errors found.</div>", unsafe_allow_html=True)

    # ---------------- Suggestions List ----------------
    st.markdown("### Suggestions")
    if errors:
        lines = []
        for w, info in errors.items():
            kind = info["kind"]
            best = info["sugs"][0] if info["sugs"] else "(no suggestion)"
            lines.append(f"{w} ({kind}) → {best}")
        st.markdown("<div class='box'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='box'>No suggestions.</div>", unsafe_allow_html=True)

    # ---------------- Final Output ----------------
    st.markdown("### Final Output")
    if st.button("Generate Final Output"):
        final = apply_corrections(st.session_state.text, errors)
        st.markdown(
            f"<div class='box' style='background:#d4f8d4; color:black; font-size:18px;'>{final}</div>",
            unsafe_allow_html=True,
        )

with right:
    st.markdown("### Dictionary (Business Corpus)")
    search = st.text_input("Search dictionary:")
    if search:
        filtered = [w for w in dictionary_words if search.lower() in w]
    else:
        filtered = dictionary_words[:4000]

    st.markdown(
        f"<div class='box' style='height:520px; overflow-y:auto;'>{'<br>'.join(filtered)}</div>",
        unsafe_allow_html=True,
    )