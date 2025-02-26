import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Define file paths
RAW_DATA_PATH = "./data/raw/bbc_sentences_raw.csv"
PROCESSED_DATA_PATH = "./data/processed/"
TFIDF_MODEL_PATH = os.path.join(PROCESSED_DATA_PATH, "tfidf_vectorizer.pkl")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_PATH, "bbc_sentences_cleaned.csv")

# Ensure processed data folder exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Load raw data
df = pd.read_csv(RAW_DATA_PATH)

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """ Function to clean and preprocess text """
    if pd.isna(text) or not isinstance(text, str):
        return ""  # Handle empty or non-string data

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]  # Remove stopwords & short words

    return " ".join(tokens)

# Apply preprocessing
df["cleaned_sentence"] = df["sentence"].astype(str).apply(clean_text)

# **Check if we have any non-empty cleaned sentences**
df = df[df["cleaned_sentence"].str.strip() != ""]

# Check if the dataset is empty after cleaning
if df.empty:
    raise ValueError("All sentences were removed during preprocessing. Check stopword filtering.")

# Apply TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 words
X_tfidf = vectorizer.fit_transform(df["cleaned_sentence"])

# Save TF-IDF model
joblib.dump(vectorizer, TFIDF_MODEL_PATH)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Preprocessing complete. TF-IDF model saved to {TFIDF_MODEL_PATH}")
print(f"✅ Cleaned data saved to {OUTPUT_FILE}")
