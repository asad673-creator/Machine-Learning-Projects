import re
import string
import warnings
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            padding-top: 1.5rem;
        }

        .hero {
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,.12);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
        }

        .hero p {
            margin-top: .65rem;
            color: #d1d5db;
            font-size: 1.05rem;
        }

        .result-card {
            padding: 1.4rem;
            border-radius: 18px;
            margin-top: 1rem;
            text-align: center;
            border: 1px solid rgba(128,128,128,.25);
        }

        .safe {
            background: rgba(34, 197, 94, .10);
            border-color: rgba(34, 197, 94, .35);
        }

        .danger {
            background: rgba(239, 68, 68, .10);
            border-color: rgba(239, 68, 68, .35);
        }

        .result-title {
            font-size: 1.65rem;
            font-weight: 700;
        }

        .metric-box {
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,.22);
            text-align: center;
        }

        .small-muted {
            color: #6b7280;
            font-size: .9rem;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 14px;
        }

        .stButton button {
            border-radius: 12px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# File paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_classifier(1).joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer(1).joblib"


# -----------------------------
# NLTK setup
# -----------------------------
@st.cache_resource
def setup_nltk():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)

    return set(stopwords.words("english")), WordNetLemmatizer()


# -----------------------------
# Load model + vectorizer
# -----------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


try:
    STOP_WORDS, LEMMATIZER = setup_nltk()
    MODEL, VECTORIZER = load_artifacts()
except Exception as e:
    st.error("The application could not load the required model files.")
    st.code(str(e))
    st.stop()


# -----------------------------
# Same preprocessing used by
# the training notebook
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    words = text.split()
    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


def predict_message(message: str):
    cleaned = clean_text(message)
    vector = VECTORIZER.transform([cleaned])
    prediction = int(MODEL.predict(vector)[0])

    # LinearSVC does not provide probabilities.
    # This converts the decision score into an easy-to-read
    # approximate confidence indicator.
    decision_score = float(MODEL.decision_function(vector)[0])
    score_confidence = 1 / (1 + __import__("math").exp(-abs(decision_score)))

    return prediction, cleaned, decision_score, score_confidence


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🛡️ SMS Spam Detector")
    st.markdown(
        "A machine-learning web app that classifies SMS messages as "
        "**Ham** or **Spam**."
    )

    st.divider()

    st.markdown("### Model")
    st.write("**Algorithm:** Linear SVC")
    st.write("**Features:** TF-IDF")
    st.write("**N-grams:** 1–2")
    st.write("**Max features:** 20,000")

    st.divider()

    st.markdown("### Preprocessing")
    st.write("✓ Lowercase")
    st.write("✓ Remove punctuation")
    st.write("✓ Remove digits")
    st.write("✓ Remove English stopwords")
    st.write("✓ Lemmatization")

    st.divider()

    st.caption("Trained on the SMS Spam Detection pipeline provided with this project.")


# -----------------------------
# Main UI
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SMS Spam Detector</h1>
        <p>Paste an SMS message below and let the trained ML model classify it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Enter your message")

message = st.text_area(
    "SMS message",
    placeholder="Example: Congratulations! You have won a free prize. Claim now!",
    height=170,
    label_visibility="collapsed",
)

col1, col2 = st.columns([3, 1])

with col1:
    predict_clicked = st.button(
        "🔍 Check Message",
        type="primary",
        use_container_width=True,
    )

with col2:
    clear_clicked = st.button(
        "Clear",
        use_container_width=True,
    )

if clear_clicked:
    st.rerun()

if predict_clicked:
    if not message.strip():
        st.warning("Please enter an SMS message first.")
    else:
        prediction, cleaned, decision_score, confidence = predict_message(message)

        if prediction == 1:
            st.markdown(
                """
                <div class="result-card danger">
                    <div class="result-title">🚨 SPAM MESSAGE</div>
                    <div>Be careful with links, offers, and requests for personal information.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card safe">
                    <div class="result-title">✅ HAM / NOT SPAM</div>
                    <div>This message was classified as a normal message.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <b>Prediction</b><br>
                    {"SPAM" if prediction == 1 else "HAM"}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <b>Message Length</b><br>
                    {len(message)} characters
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <b>Model Indicator</b><br>
                    {confidence:.1%}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View preprocessing"):
            st.write("**Original message:**")
            st.write(message)

            st.write("**Cleaned message:**")
            st.code(cleaned or "(empty after preprocessing)")

        with st.expander("View model score"):
            st.write(
                "The classifier is a Linear SVC, so it does not output a true "
                "probability. The value below is the raw decision score."
            )
            st.metric("Decision score", f"{decision_score:.4f}")


# -----------------------------
# Example messages
# -----------------------------
st.divider()
st.markdown("### 💡 Try an example")

examples = {
    "Normal SMS": "Hey, are we still meeting at 5 today?",
    "Spam SMS": "Congratulations! You have won a free prize. Claim now!",
    "Promotion": "FREE entry to win a cash prize. Text WIN to claim your reward!",
}

example_cols = st.columns(3)

for column, (label, example) in zip(example_cols, examples.items()):
    with column:
        if st.button(label, use_container_width=True):
            st.session_state["example_message"] = example
            st.rerun()

if "example_message" in st.session_state:
    st.info(f"Example: {st.session_state['example_message']}")
    st.caption("Copy the example into the message box above and click **Check Message**.")
