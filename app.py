import re
import urllib.parse
from spellchecker import SpellChecker
import spacy
import streamlit as st
from collections import Counter
import math

# ----------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------
st.set_page_config(page_title="Spelling Correction System", layout="wide")

# ----------------------------------------------------
# Desktop-style CSS (JetBrains Darcula + Wavy Underlines)
# ----------------------------------------------------
st.markdown("""
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

/* GLOBAL */
body, .main {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    padding: 10px;
    background-color: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
}
.stButton > button:hover {
    background-color: #6897BB;
    border-color: #6897BB;
    color: black;
}

/* Inputs */
textarea, input, .stTextInput > div > div > input {
    background-color: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* Boxes */
.box {
    background-color: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
}

/* Dictionary */
.dict-box {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    height: 540px;
    padding: 10px;
    overflow-y: scroll;
}

/* Wavy Underlines */
@keyframes squiggly {
    from { background-position-x: 0; }
    to { background-position-x: -20px; }
}

span.underline {
    padding-bottom: 2px;
}

span.nonword {
    background-image: repeating-linear-gradient(
        -45deg,
        transparent 0 3px,
        var(--err) 3px 6px
    );
    background-size: 6px 6px;
    background-repeat: repeat-x;
    animation: squiggly 1s linear infinite;
}

span.realword {
    background-image: repeating-linear-gradient(
        -45deg,
        transparent 0 3px,
        var(--real) 3px 6px
    );
    background-size: 6px 6px;
    background-repeat: repeat-x;
    animation: squiggly 1s linear infinite;
}

span.grammar {
    background-image: repeating-linear-gradient(
        -45deg,
        transparent 0 3px,
        var(--grammar) 3px 6px
    );
    background-size: 6px 6px;
    background-repeat: repeat-x;
    animation: squiggly 1s linear infinite;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Bigram Model (Norvig's big.txt)
# ----------------------------------------------------
@st.cache_resource
def load_bigram_model():
    with open("big.txt", "r", encoding="utf8") as f:
        text = f.read().lower()

    words = re.findall(r"\w+", text)
    unigrams = Counter(words)
    bigrams = Counter(zip(words, words[1:]))
    total = sum(unigrams.values())

    return unigrams, bigrams, total

unigrams, bigrams, total_words = load_bigram_model()

def bigram_prob(prev, word):
    return (bigrams[(prev, word)] + 1) / (unigrams[prev] + len(unigrams))

def sentence_score(words):
    score = 0
    for i in range(1, len(words)):
        score += math.log(bigram_prob(words[i-1], words[i]))
    return score

# ----------------------------------------------------
# SpellChecker + clean dictionary
# ----------------------------------------------------
@st.cache_resource
def load_resources():
    nlp = spacy.load("en_core_web_sm")
    spell = SpellChecker()

    clean_vocab = [
        w for w in spell.word_frequency.words()
        if w.isalpha() and w.islower()
    ]

    return nlp, spell, sorted(clean_vocab)

nlp, spell, dictionary_words = load_resources()

# ----------------------------------------------------
# Semantic Rules (simple but effective)
# ----------------------------------------------------
VALID_OBJECTS = {
    "eat": {"food", "meal", "rice", "burger", "apple", "bread"},
    "drink": {"water", "tea", "coffee", "juice", "milk"},
    "read": {"book", "article", "report", "document", "paper"},
    "write": {"report", "notes", "paper", "essay"},
    "take": {"medicine", "pill", "tablet", "drug"},
}

SEMANTIC_MAP = {
    "book": "book",
    "novel": "book",

    "rice": "rice",
    "nasi": "rice",
    "goreng": "rice",

    "water": "water",
    "tea": "water",
    "coffee": "water",

    "medicine": "medicine",
    "tablet": "medicine",
}

# ----------------------------------------------------
# Edit Distance
# ----------------------------------------------------
def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[n][m]

# ----------------------------------------------------
# Ranked Suggestions
# ----------------------------------------------------
def ranked_suggestions(word, n=3):
    cands = spell.candidates(word)
    cands = {c for c in cands if c.lower() != word.lower()}

    def score(c):
        return (edit_distance(word, c), c)

    return sorted(cands, key=score)[:n]

# ----------------------------------------------------
# Replace helper
# ----------------------------------------------------
def replace_word_in_text(text, old, new):
    return re.sub(rf"\b{old}\b", new, text, flags=re.IGNORECASE)

# URL replacement
params = st.query_params
if "replace" in params:
    old, new = params["replace"].split("|")
    old = urllib.parse.unquote(old)
    new = urllib.parse.unquote(new)
    st.session_state.text = replace_word_in_text(st.session_state.get("text", ""), old, new)
    st.query_params.clear()
    st.rerun()

# ----------------------------------------------------
# MAIN Detection Logic (Hybrid System)
# ----------------------------------------------------
def check_text(text):
    doc = nlp(text)
    tokens = [t for t in doc if t.is_alpha]

    errors = {}

    # 1️⃣ Non-word errors
    unknown = spell.unknown([t.text.lower() for t in tokens])
    for t in tokens:
        lw = t.text.lower()
        if lw in unknown:
            sugg = ranked_suggestions(lw)
            if sugg:
                errors[lw] = (sugg, "non-word")

    # 2️⃣ Real-word errors (bigram/noisy channel)
    for t in tokens:
        lw = t.text.lower()
        if lw in errors: continue
        if lw in spell:
            suggs = ranked_suggestions(lw, 3)
            best_score = float("-inf")
            best = None

            original = [w.text.lower() for w in doc]
            original_score = sentence_score(original)

            for s in suggs:
                temp = list(original)
                temp[original.index(lw)] = s
                sc = sentence_score(temp)
                if sc > best_score:
                    best_score = sc
                    best = s

            if best and best_score > original_score:
                errors[lw] = ([best], "real-word")

    # 3️⃣ Semantic verb–object errors
    for i, t in enumerate(tokens[:-1]):
        verb = t.text.lower()
        obj = tokens[i+1].text.lower()

        if verb in VALID_OBJECTS:
            obj_cat = SEMANTIC_MAP.get(obj, obj)
            if obj_cat not in VALID_OBJECTS[verb]:
                errors[verb] = ([f"(semantic) maybe replace"], "real-word")

    # 4️⃣ Grammar (subject-verb)
    singular = {"he", "she", "it", "this", "that"}
    plural = {"they", "we", "i", "you", "these", "those"}

    for tok in doc:
        if tok.tag_ in ("VBP", "VBZ") and tok.dep_ == "ROOT":
            for child in tok.children:
                if child.dep_ == "nsubj":
                    subj = child.text.lower()
                    if subj in singular and tok.tag_ == "VBP":
                        errors[tok.text.lower()] = ([tok.lemma_ + "s"], "grammar")
                    if subj in plural and tok.tag_ == "VBZ":
                        errors[tok.text.lower()] = ([tok.lemma_],
 "grammar")

    return errors

# ----------------------------------------------------
# Highlight Output
# ----------------------------------------------------
def highlight(text, errors):
    tokens = re.findall(r"\w+|[^\w\s]", text)
    out = []

    for tok in tokens:
        lw = tok.lower()
        if lw in errors:
            k = errors[lw][1]
            cls = "nonword" if k == "non-word" else "grammar" if k=="grammar" else "realword"
            out.append(f"<span class='{cls} underline'>{tok}</span>")
        else:
            out.append(tok)

    return " ".join(out)

# ----------------------------------------------------
# UI Layout
# ----------------------------------------------------
left, right = st.columns([2.5, 1])

with left:
    st.markdown("<h2>Spelling Correction System</h2>", unsafe_allow_html=True)

    check_btn = st.button("Check Text")
    clear_btn = st.button("Clear")

    if "text" not in st.session_state:
        st.session_state.text = ""

    if clear_btn:
        st.session_state.text = ""

    text = st.text_area("", st.session_state.text, height=260)
    st.session_state.text = text[:500]

    errors = check_btn and check_text(text) or {}

    st.markdown("### Suggestions")
    if not errors:
        st.markdown("<div class='box'>No errors found.</div>", unsafe_allow_html=True)
    else:
        out = "<div class='box'>"
        for w, (sg, typ) in errors.items():
            out += f"{w} ({typ}) → "
            links = []
            for s in sg:
                links.append(
                    f"<a href='?replace={urllib.parse.quote(w)}|{urllib.parse.quote(s)}'>{s}</a>"
                )
            out += ", ".join(links) + "<br>"
        out += "</div>"
        st.markdown(out, unsafe_allow_html=True)

    if errors:
        st.markdown("### Highlighted Text")
        st.markdown(
            f"<div class='box'>{highlight(text, errors)}</div>",
            unsafe_allow_html=True
        )

with right:
    st.markdown("### Dictionary")

    search = st.text_input("", "")
    filtered = [w for w in dictionary_words if search.lower() in w] if search else dictionary_words[:4000]
    st.markdown(f"<div class='dict-box'>{'<br>'.join(filtered)}</div>", unsafe_allow_html=True)
