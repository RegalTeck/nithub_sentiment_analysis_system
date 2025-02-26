  from fastapi import FastAPI
  from app.api.v1.sentiment import router as sentiment_router

  app = FastAPI(
      title="Sentiment Analysis API",
      version="1.0.0",
      description="API for sentiment analysis using a trained model."
  )

  app.include_router(sentiment_router, prefix="/api/v1")