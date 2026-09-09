# SMS Spam Detector — Streamlit

A Streamlit frontend for the trained SMS spam classification model.

## Files

Keep these files in the same folder:

- `app.py`
- `spam_classifier(1).joblib`
- `tfidf_vectorizer(1).joblib`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app uses the same text preprocessing described in the training notebook:
lowercase → remove punctuation → remove digits → remove English stopwords → lemmatize.

The saved model is a Linear SVC and the saved vectorizer is TF-IDF with unigram/bigram features.
