import re
import streamlit as st
from spellchecker import SpellChecker
import spacy

# ----------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------
st.set_page_config(page_title="Spelling Correction System", layout="wide")

# ----------------------------------------------------
# JetBrains-style Dark UI + Wavy Underlines
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

body, .main {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Segoe UI', sans-serif;
}

h2 {
    color: var(--text) !important;
    font-weight: 600 !important;
}

.box {
    background-color: var(--card);
    border: 1px solid var(--border);
    padding: 12px;
    border-radius: 8px;
    margin-top: 10px;
}

/* Wavy underline animation */
@keyframes squiggly {
    from { background-position-x: 0; }
    to { background-position-x: -20px; }
}

span.nonword, span.realword, span.grammar {
    background-size: 6px 6px;
    background-repeat: repeat-x;
    background-position-y: bottom;
    animation: squiggly 1s linear infinite;
    padding-bottom: 1px;
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
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load NLP + SpellChecker
# ----------------------------------------------------
@st.cache_resource
def load_resources():
    nlp = spacy.load("en_core_web_sm")
    spell = SpellChecker()

    # Clean dictionary for optional sidebar (only alphabetic lowercase words)
    dict_words = sorted(
        w for w in spell.word_frequency.words()
        if w.isalpha() and w.islower()
    )
    return nlp, spell, dict_words

nlp, spell, dictionary_words = load_resources()

# ----------------------------------------------------
# Small Confusion Sets (Real-word spelling)
# ----------------------------------------------------
CONFUSION_SETS = {
    "sea": ["see"],
    "see": ["sea"],
    "form": ["from"],
    "from": ["form"],
    "there": ["their", "they're"],
    "their": ["there", "they're"],
    "they're": ["their", "there"],
    "to": ["too", "two"],
    "too": ["to", "two"],
    "two": ["to", "too"],
}

# ----------------------------------------------------
# Simple Semantic Rules (light verb–object check)
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
    "juice": "water",
    "milk": "water",

    "medicine": "medicine",
    "tablet": "medicine",
    "pill": "medicine",
    "drug": "medicine",
}

# ----------------------------------------------------
# Levenshtein Edit Distance
# ----------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[n][m]

def ranked_suggestions(word: str, max_suggestions: int = 3):
    candidates = spell.candidates(word)
    candidates = {c for c in candidates if c.lower() != word.lower()}
    if not candidates:
        return []
    return sorted(candidates, key=lambda c: (edit_distance(word, c), c))[:max_suggestions]

# ----------------------------------------------------
# Core Detection Logic
# ----------------------------------------------------
def check_text(text: str):
    doc = nlp(text)
    tokens = [t for t in doc if t.is_alpha]
    lower_tokens = [t.text.lower() for t in tokens]

    errors = {}  # word(lower) -> {"kind": ..., "sugs": [...]}

    # 1) Non-word spelling errors (SpellChecker)
    unknown = spell.unknown(lower_tokens)
    for tok in tokens:
        lw = tok.text.lower()
        if lw in unknown:
            sugs = ranked_suggestions(lw)
            if not sugs:
                sugs = ["(no suggestion)"]
            errors[lw] = {"kind": "non-word", "sugs": sugs}

    # 2) Real-word confusion set errors (sea/see, there/their...)
    for tok in tokens:
        lw = tok.text.lower()
        if lw in errors:
            continue
        if lw in CONFUSION_SETS:
            # Very simple: show alternative forms
            sugs = CONFUSION_SETS[lw]
            errors[lw] = {"kind": "real-word", "sugs": sugs}

    # 3) Improved Semantic Checking (verb-object meaning)

    SEMANTIC_REPLACEMENTS = {
        "eat": ["read", "take", "consume"],
        "drink": ["eat", "taste"],
        "read": ["review", "study"],
        "write": ["draft", "compose"],
        "take": ["eat", "consume"],
    }

    for tok in doc:

        if tok.pos_ != "VERB":
            continue

        verb_lemma = tok.lemma_.lower()
        verb_form = tok.text.lower()   # << ACTUAL SURFACE WORD (e.g., eating)

        if verb_lemma not in VALID_OBJECTS:
            continue

        # find object
        obj_token = None
        for child in tok.children:
            if child.dep_ in ("dobj", "obj") and child.is_alpha:
                obj_token = child
                break

        if not obj_token:
            continue

        obj_lemma = obj_token.lemma_.lower()
        obj_form = obj_token.text.lower()  # << ACTUAL SURFACE WORD (e.g., book)

        obj_cat = SEMANTIC_MAP.get(obj_lemma, obj_lemma)

        if obj_cat not in VALID_OBJECTS[verb_lemma]:

            # get appropriate replacement verbs
            suggestions = SEMANTIC_REPLACEMENTS.get(verb_lemma, ["Check meaning"])

            # highlight BOTH actual words, not lemma
            for w in (verb_form, obj_form):
                if w not in errors:
                    errors[w] = {
                        "kind": "real-word",
                        "sugs": suggestions
                    }



    # 4) Basic grammar: subject–verb agreement
    singular = {"he", "she", "it", "this", "that"}
    plural = {"they", "we", "i", "you", "these", "those"}

    for tok in doc:
        if tok.tag_ in ("VBP", "VBZ") and tok.dep_ == "ROOT":
            for child in tok.children:
                if child.dep_ == "nsubj" and child.is_alpha:
                    subj = child.text.lower()
                    lw = tok.text.lower()
                    if lw in errors:
                        continue
                    if subj in singular and tok.tag_ == "VBP":
                        errors[lw] = {"kind": "grammar", "sugs": [tok.lemma_ + "s"]}
                    elif subj in plural and tok.tag_ == "VBZ":
                        errors[lw] = {"kind": "grammar", "sugs": [tok.lemma_]}

    return errors

# ----------------------------------------------------
# Highlight text based on errors
# ----------------------------------------------------
def highlight(text, errors):
    words = text.split()
    out = []

    for w in words:
        lw = re.sub(r"\W+", "", w.lower())
        if lw in errors:
            kind = errors[lw]["kind"]
            css = "nonword" if kind == "non-word" else "grammar" if kind == "grammar" else "realword"
            out.append(f"<span class='{css}'>{w}</span>")
        else:
            out.append(w)

    return " ".join(out)


# ----------------------------------------------------
# Apply chosen corrections (for preview/final output)
# ----------------------------------------------------
def apply_corrections(original_text: str, corrections: dict):
    parts = re.findall(r"\w+|\W+", original_text)
    out = []
    for p in parts:
        lw = p.lower()
        if lw in corrections and corrections[lw] not in ("(no change)", "(no suggestion)"):
            out.append(corrections[lw])
        else:
            out.append(p)
    return "".join(out)

# ----------------------------------------------------
# UI LAYOUT (Dropdown + Preview + Final Output)
# ----------------------------------------------------
left, right = st.columns([2.5, 1])

with left:
    st.markdown("<h2>Spelling & Grammar Checker</h2>", unsafe_allow_html=True)

    if "text" not in st.session_state:
        st.session_state.text = ""
    if "errors" not in st.session_state:
        st.session_state.errors = {}
    if "corrections" not in st.session_state:
        st.session_state.corrections = {}

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        check_btn = st.button("Check Text")
    with col_btn2:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.session_state.text = ""
        st.session_state.errors = {}
        st.session_state.corrections = {}

    text = st.text_area(
        "Main Text",
        st.session_state.text,
        height=220
    )
    st.session_state.text = text

    # Run checker when button pressed
    if check_btn:
        errs = check_text(text)
        st.session_state.errors = errs
        st.session_state.corrections = {w: "(no change)" for w in errs.keys()}

    errors = st.session_state.errors
    corrections = st.session_state.corrections

    # Suggestions with dropdowns
    st.markdown("### Suggestions (Dropdown per Error)")
    if not errors:
        st.markdown("<div class='box'>No errors found.</div>", unsafe_allow_html=True)
    else:
        with st.container():
            for w, info in errors.items():
                sugs = info["sugs"]
                kind = info["kind"]
                label = f"Choose replacement for '{w}' ({kind})"
                options = ["(no change)"] + sugs
                # key per word
                selected = st.selectbox(label, options, key=f"sel_{w}")
                corrections[w] = selected
            st.session_state.corrections = corrections

    # Preview sentence
    st.markdown("### Sentence Preview")
    if errors:
        preview = apply_corrections(text, corrections)
        st.markdown(f"<div class='box'>{preview}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='box'>Nothing to preview.</div>", unsafe_allow_html=True)

    # Final corrected output
    st.markdown("### Final Corrected Output")
    if st.button("Generate Final Output"):
        final_output = apply_corrections(text, corrections)
        st.markdown(
            f"<div class='box' style='background:#d4f8d4; color:black;'>{final_output}</div>",
            unsafe_allow_html=True
        )

    # Highlight panel
    st.markdown("### Highlighted Text")
    if errors:
        highlighted = highlight(text, errors)
        st.markdown(f"<div class='box'>{highlighted}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='box'>No highlights.</div>", unsafe_allow_html=True)

with right:
    st.markdown("### Dictionary (from SpellChecker)")
    search = st.text_input("Search dictionary:", "")
    if search:
        filtered = [w for w in dictionary_words if search.lower() in w]
    else:
        filtered = dictionary_words[:4000]
    st.markdown(f"<div class='box' style='height:480px; overflow-y:auto;'>{'<br>'.join(filtered)}</div>", unsafe_allow_html=True)
