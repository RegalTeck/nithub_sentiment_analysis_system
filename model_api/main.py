# model_api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from fastapi.staticfiles import StaticFiles

# Define the request and response models
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

# Initialize the FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="API for sentiment analysis using a trained model."
)

# Load the pre-trained model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), '../models/sentiment_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '../models/tfidf_vectorizer.pkl')

with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = joblib.load(vectorizer_file)

@app.post("/analyze", response_model=SentimentResponse)
async def analyze(request: SentimentRequest):
    """
    Analyze the sentiment of the given text.
    """
    # Transform the input text using the loaded TF-IDF vectorizer
    transformed_text = vectorizer.transform([request.text])

    # Predict the sentiment
    prediction = model.predict(transformed_text)[0]
    confidence = max(model.predict_proba(transformed_text)[0])

    return SentimentResponse(sentiment=prediction, confidence=confidence)

# Add a root route
@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API"}

#Mount static folder if you want to support favicon.ico or other static files.
#app.mount("/static", StaticFiles(directory="static"), name="static")