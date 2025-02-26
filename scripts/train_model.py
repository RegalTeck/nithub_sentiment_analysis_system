import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import joblib

# Load processed data
data_path = "data/processed/bbc_sentences_cleaned.csv"
df = pd.read_csv(data_path)

# ✅ Improved Sentiment Labeling (TextBlob polarity)
def label_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

# Apply sentiment labeling
df["sentiment"] = df["cleaned_sentence"].apply(label_sentiment)

# Save labeled dataset
df.to_csv("data/processed/bbc_sentences_labeled.csv", index=False)
print("✅ Sentiment-labeled dataset saved: data/processed/bbc_sentences_labeled.csv")

# ✅ Increase TF-IDF Features (from 1000 to 5000)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_sentence"])
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Use RandomForestClassifier Instead of Logistic Regression
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# ✅ Hyperparameter Tuning with GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Train Best Model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Improved Model Accuracy: {accuracy:.2f}")

# Save Model and Vectorizer
joblib.dump(best_model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Model saved: models/sentiment_model.pkl")
print("✅ Vectorizer saved: models/tfidf_vectorizer.pkl")
