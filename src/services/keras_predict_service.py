from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

# Define the 'src' directory path
SRC_DIR = Path(__file__).resolve().parent.parent

# Define the path to the models directory inside 'src'
MODELS_DIR = SRC_DIR / "models"

# Define full paths to the saved model and pipeline files
MODEL_PATH = MODELS_DIR / "final_model.keras"
PIPELINE_PATH = MODELS_DIR / "final_pipeline.pkl"

# Load the preprocessing pipeline and the Keras model
pipeline = joblib.load(PIPELINE_PATH)
model = load_model(MODEL_PATH)

def predict_keras(text: str) -> int:
    """
    Predicts the class (0 or 1) for the given text using the loaded pipeline and model.
    Returns 1 if predicted probability > 0.5, else 0.
    """
    # Transform the input text using the preprocessing pipeline
    processed = pipeline.transform([text])
    # Predict probability using the Keras model
    prob = model.predict(processed)[0][0]
    # Convert probability to binary class
    return int(prob > 0.5)

if __name__ == "__main__":
    sample_text = "Explosion in downtown area, people are injured!"
    prediction = predict_keras(sample_text)
    print(f"Prediction: {prediction}")
