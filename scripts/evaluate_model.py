import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load Model and Vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load Test Data
df = pd.read_csv("data/processed/bbc_sentences_labeled.csv")

# Transform Text Using TF-IDF
X_test = vectorizer.transform(df["cleaned_sentence"])
y_test = df["sentiment"]

# Predict Sentiments
y_pred = model.predict(X_test)

# Classification Report
print("\n📊 **Classification Report:**\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
