import joblib
from tensorflow.keras.models import load_model

# Load pipeline and model once
pipeline = joblib.load("src/models/final_pipeline.pkl")
model = load_model("src/models/final_model.keras")

def predict_keras(text: str) -> int:
    processed = pipeline.transform([text])
    prob = model.predict(processed)[0][0]
    return int(prob > 0.5)

if __name__ == "__main__":
    sample_text = "Explosion in downtown area, people are injured!"
    prediction = predict_keras(sample_text)
    print(f"Prediction: {prediction}")