# Tweet Classifier API

This is a FastAPI service that classifies tweets as **real disasters** or **not**, using both:
- A fine-tuned **Keras** model (TF-IDF + Neural Network)
- **GPT-4-mini** via OpenAI API

Built with **FastAPI**, this service allows:
- Training a neural network model (`/train`)
- Classifying tweets (`/predict`)

##  Project Structure

- `data/` — contains raw data  
  - `train.csv`
- `logs/` — stores application logs  
  - `app.log`
- `src/` — main source code  
  - `app.py` — FastAPI app  
  - `schemas.py` — request/response models  
  - `models/` — saved model and pipeline  
    - `final_model.keras`  
    - `final_pipeline.pkl`  
  - `services/` — logic for training & prediction  
    - `train_service.py`  
    - `gpt_predict_service.py`  
    - `keras_predict_service.py`  
  - `utils/` — helper functions  
    - `logger.py`  
    - `pipeline.py`  
    - `text_cleaner.py`
- `run.py` — script to run the app locally
- `.env` — environment variables  
- `.env.example` — example environment file for setup  
- `.gitignore` — ignored files  
- `requirements.txt` — dependencies  
- `README.md` — you're reading it!

##  Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/anastasia-vishnevsky/tweet-classifier-api.git
cd tweet-classifier-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App

To start the FastAPI server locally:

```bash
python run.py
```
Then open your browser and go to:
- Swagger UI: http://127.0.0.1:8000/docs ←  for training & prediction
- Root check: http://127.0.0.1:8000/

## Training the Model

To train the Keras model, open http://127.0.0.1:8000/docs and:
- Find the POST /train endpoint.
- Click “Try it out”.
- Hit Execute.
- View training metrics in the response and logs in logs/app.log.

The model will be saved to:
- src/models/final_model.keras
- src/models/final_pipeline.pkl

## Classifying Tweets

- In http://127.0.0.1:8000/docs, find POST /predict.
- Click “Try it out”.
- Enter your tweet in the JSON field, for example:
```json
{
  "text": "Explosion in downtown area, many injured"
}
```
- Click Execute.
- See the GPT and Keras predictions in the response.