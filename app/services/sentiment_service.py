  import pickle
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model import LogisticRegression
  import os

  # Load the TF-IDF vectorizer and the trained model
  model_path = os.path.join(os.path.dirname(__file__), '../../models/sentiment_model.pkl')
  vectorizer_path = os.path.join(os.path.dirname(__file__), '../../models/tfidf_vectorizer.pkl')

  with open(model_path, 'rb') as model_file:
      model = pickle.load(model_file)

  with open(vectorizer_path, 'rb') as vectorizer_file:
      vectorizer = pickle.load(vectorizer_file)

  def analyze_sentiment(text: str):
      """
      Analyze the sentiment of the given text.
      Returns the sentiment label and the confidence score.
      """
      # Transform the text using the loaded TF-IDF vectorizer
      transformed_text = vectorizer.transform([text])

      # Predict the sentiment
      prediction = model.predict(transformed_text)[0]
      confidence = max(model.predict_proba(transformed_text)[0])

      return prediction, confidence
