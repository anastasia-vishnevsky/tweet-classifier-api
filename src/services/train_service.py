from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score

# Import pipeline creator
from src.utils.pipeline import create_pipeline

# Define base directories
SRC_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = SRC_DIR / "models"
DATA_DIR = SRC_DIR.parent / "data"

MODEL_PATH = MODELS_DIR / "final_model.keras"
PIPELINE_PATH = MODELS_DIR / "final_pipeline.pkl"
CSV_PATH = DATA_DIR / "train.csv"

def train_model(csv_path=CSV_PATH):
    # Load data
    df = pd.read_csv(csv_path)
    df = df[["text", "target"]]

    # Create pipeline
    pipeline = create_pipeline()

    # Split data
    X = df['text']
    y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Transform data
    X_train_tfidf = pipeline.fit_transform(X_train).toarray()
    X_val_tfidf = pipeline.transform(X_val).toarray()

    # Define the model
    model = Sequential([
        Input(shape=(X_train_tfidf.shape[1],)),
        Dense(24, activation='relu', kernel_regularizer=l2(0.02)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_tfidf, y_train,
        validation_data=(X_val_tfidf, y_val),
        epochs=60,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_pred_probs = model.predict(X_val_tfidf)
    y_pred = (y_pred_probs > 0.5).astype(int)

    f1 = f1_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save model and pipeline
    model.save(MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)

    return {"f1": f1, "accuracy": accuracy, "message": "Model trained and saved successfully."}

if __name__ == "__main__":
    train_model()
