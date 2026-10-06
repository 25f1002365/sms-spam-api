# SMS Spam Detector API

A small machine learning API that reads an SMS message and tells you whether it is **spam** or **not spam**.

Built for the *Getting Started with ML in Production* workshop assignment:
**Model → FastAPI → GitHub → Render → Logging & Monitoring**

**Live API:** `https://<your-app>.onrender.com`  (docs: `/docs`)

## What the model predicts

Given the text of an SMS, the model predicts one of two classes:

| prediction | label      |
|-----------|------------|
| 0         | not spam   |
| 1         | spam       |

- **Data:** the public SMS Spam Collection (5,574 labelled messages, `data/sms.tsv`)
- **Model:** scikit-learn `Pipeline` = TF-IDF (1-2 word n-grams) + Logistic Regression
- **Test accuracy:** about 97.7%
- The whole pipeline is saved as one file, `model.pkl`, and loaded once when the API starts.

## Endpoints

| Method | Path       | What it does                                    |
|--------|-----------|--------------------------------------------------|
| GET    | `/health`  | Returns API status and whether the model loaded |
| POST   | `/predict` | Takes a message, returns the prediction         |

### Example request body for `/predict`

```json
{
  "message": "WINNER!! You have won a free prize. Call 09061701461 now to claim"
}
```

### Example response

```json
{
  "prediction": 1,
  "label": "spam",
  "spam_probability": 0.9659
}
```

### Example with curl

```bash
curl -X POST https://<your-app>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Hey, are we still meeting for lunch tomorrow?"}'
```

An invalid body (for example `{"text": 123}` or an empty message) returns **422**.

## How to run it locally

```bash
git clone https://github.com/<your-username>/sms-spam-api.git
cd sms-spam-api
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs and use "Try it out".

To retrain the model (optional):

```bash
python train.py
```

## Logging

Every request is logged as one JSON line (method, path, status code, latency in ms),
visible in the Render **Logs** tab.

## Project files

```
main.py            FastAPI app (/health, /predict, logging middleware)
train.py           Trains the model and exports model.pkl
model.pkl          Trained model
requirements.txt   Dependencies
data/sms.tsv       Training data
```
