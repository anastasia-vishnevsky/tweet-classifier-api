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
- `tests/` — unit tests for core logic
  - `test_text_cleaner.py` — tests for the TextCleaner class
- `run.py` — script to run the app locally
- `.env` — environment variables  
- `.env.example` — example environment file for setup 
- `Dockerfile` — builds a container image for the app
- `docker-compose.yml` — defines and runs the container app
- `.gitignore` — ignored files  
- `requirements.txt` — dependencies  
- `README.md` — you're reading it!

##  Installation

1. Clone the repository:

```bash
git clone https://github.com/anastasia-vishnevsky/tweet-classifier-api.git
cd tweet-classifier-api
````

2. Create and activate a virtual environment:

- On **macOS/Linux**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- On **Windows**:
```bash
python -m venv .venv
.venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file or copy the provided `.env.example`. You’ll need to include:
OPENAI_API_KEY=your_openai_key_here

> Make sure the `.env` file is located in the root of your project. It is automatically loaded using `python-dotenv`.

## Running the App

To start the FastAPI server **locally**:

```bash
python run.py
```
Then open your browser and go to:
- Swagger UI: http://127.0.0.1:8000/docs ←  for training & prediction
- Root check: http://127.0.0.1:8000/

Еo start the FastAPI server in **Docker**:

```bash
docker compose up --build
```
Then open your browser and go to:
- Swagger UI: http://localhost:8000/docs ←  for training & prediction
- Root check: http://localhost:8000/

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

## Running tests

Unit tests are located in the tests/ folder and use pytest.

To run tests **locally**:
```bash
python -m pytest
```
If you’re using **Docker**, tests will run automatically when the container launches.


