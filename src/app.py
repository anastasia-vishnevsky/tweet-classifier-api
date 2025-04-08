from fastapi import FastAPI
from dotenv import load_dotenv
import os
from src.schemas import Tweet
from src.services.gpt_predict_service import predict_gpt
from src.services.keras_predict_service import predict_keras
from src.services.train_service import train_model
from src.utils.logger import logger

# Load environment variables
load_dotenv()

# Initialize app
app = FastAPI()
logger.info("FastAPI app started")

@app.get("/")
def home():
    logger.info("GET / called.")
    return {
        "title": "Tweet Classifier",
        "description": "This API uses GPT and Keras to classify tweets as real disasters or not",
        "endpoints": {
            "/predict": "POST: Classify a tweet",
            "/train": "POST: Train the model",
            "/docs": "Interactive Swagger UI"
        }
    }

@app.post("/train")
def train():
    logger.info("Training initiated via /train")
    result = train_model()
    logger.info(f"Training completed. Accuracy: {result['accuracy']:.4f}, F1 Score: {result['f1']:.4f}")
    return result

@app.post("/predict")
def predict(tweet: Tweet):
    logger.info(f'Prediction requested → "{tweet.text}"')
    # GPT prediction
    gpt_prediction = predict_gpt(tweet.text)
    # Keras prediction
    keras_prediction = predict_keras(tweet.text)
    logger.info(f'Prediction → "{tweet.text}" GPT: {gpt_prediction}, Keras: {keras_prediction}')
    return {
        "gpt_model_prediction": gpt_prediction,
        "keras_model_prediction": keras_prediction
    }