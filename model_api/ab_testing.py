from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import random
import time

app = FastAPI()

# Load your models
baseline_model = joblib.load("models/baseline_sentiment_model.pkl")
variant_model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    model_version: str

# A/B testing parameters
AB_TEST_RATIO = 0.5  # 50% traffic to each model
test_results = {"baseline": [], "variant": []}

@app.post("/analyze", response_model=SentimentResponse)
async def analyze(request: SentimentRequest):
    start_time = time.time()
    transformed_text = vectorizer.transform([request.text])

    if random.random() < AB_TEST_RATIO:
        # Use baseline model
        prediction = baseline_model.predict(transformed_text)[0]
        confidence = max(baseline_model.predict_proba(transformed_text)[0])
        model_used = "baseline"
    else:
        # Use variant model
        prediction = variant_model.predict(transformed_text)[0]
        confidence = max(variant_model.predict_proba(transformed_text)[0])
        model_used = "variant"
    end_time = time.time()
    latency = end_time - start_time

    #Store results for later analysis.
    test_results[model_used].append({"prediction": prediction, "confidence": confidence, "latency": latency})

    return SentimentResponse(sentiment=prediction, confidence=confidence, model_version=model_used)

@app.get("/ab_test_results")
async def get_test_results():
    return test_results