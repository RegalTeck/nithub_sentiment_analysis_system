  from fastapi import APIRouter
  from pydantic import BaseModel
  from app.services.sentiment_service import analyze_sentiment

  router = APIRouter()

  class SentimentRequest(BaseModel):
      text: str

  class SentimentResponse(BaseModel):
      sentiment: str
      confidence: float

  @router.post("/analyze", response_model=SentimentResponse)
  async def analyze(request: SentimentRequest):
      sentiment, confidence = analyze_sentiment(request.text)
      return SentimentResponse(sentiment=sentiment, confidence=confidence)
