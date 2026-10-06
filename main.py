"""SMS Spam Detector API (FastAPI)."""
import json
import logging
import os
import time

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spam-api")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Load the trained model once, at startup (no retraining on requests)
try:
    model = joblib.load(MODEL_PATH)
    logger.info("model loaded")
except Exception as exc:  # model missing or broken
    model = None
    logger.error("model failed to load: %s", exc)

app = FastAPI(
    title="SMS Spam Detector",
    description="Send an SMS message, get back whether it is spam or not.",
    version="1.0",
)


class Message(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, examples=["You won a free prize! Call now"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(json.dumps({
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round(latency_ms, 2),
    }))
    return response


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def home():
    """Simple web UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(body: Message):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = body.message
    pred = int(model.predict([text])[0])
    spam_prob = float(model.predict_proba([text])[0][1])
    return {
        "prediction": pred,
        "label": "spam" if pred == 1 else "not spam",
        "spam_probability": round(spam_prob, 4),
    }