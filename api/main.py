# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Define the request model
class SentimentRequest(BaseModel):
    text: str

# Initialize the FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="API for sentiment analysis using a trained model."
)

# Load the pre-trained model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), '../models/sentiment_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '../models/tfidf_vectorizer.pkl')

# Ensure the model and vectorizer files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

# Load the model and vectorizer
with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = joblib.load(vectorizer_file)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/analyze/")
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze the sentiment of the provided text.
    """
    # Transform the input text using the loaded TF-IDF vectorizer
    transformed_text = vectorizer.transform([request.text])

    # Predict the sentiment
    prediction = model.predict(transformed_text)[0]
    confidence = max(model.predict_proba(transformed_text)[0])

    return {"sentiment": prediction, "confidence": confidence}
