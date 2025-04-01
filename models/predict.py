import joblib
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained model & vectorizer
model = joblib.load("models/trained/nb_news_classifier.pkl")
vectorizer = joblib.load("models/trained/nb_tfidf_vectorizer.pkl")

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def predict_news(news_text):
    text = clean_text(news_text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return "Real News" if prediction == 1 else "Fake News"

# Test
sample_text = """
An unidentified source has reported that an alien spacecraft secretly landed in California last night. Authorities have allegedly covered up the incident to prevent public panic.
"""

print("Prediction:", predict_news(sample_text))
