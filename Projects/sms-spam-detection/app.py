import re
import string
import os
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "spam_classifier(1).joblib")
model = joblib.load(model_path)
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer(1).joblib")
vectorizer = joblib.load(vectorizer_path)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

st.title("SMS Spam Detector")
st.markdown("Enter a message to check")

message = st.text_area("SMS Message")

if st.button("Predict"):
    text = message.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    cleaned = " ".join(words)

    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.error("This message is SPAM")
    else:
        st.success("This message is NOT spam")
