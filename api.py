"""
Backend FastAPI

Chạy:
    uvicorn api:app --reload --port 8000

Production (sau Nginx/xCloud, xem DEPLOY.md):
    .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scam_detector import ScamSMSDetector
from scam_detector.config import CORS_ORIGINS

MAX_MESSAGE_LENGTH = 2000

app = FastAPI(title="Scam SMS Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

detector = ScamSMSDetector()

class PredictRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Raw SMS content to check",
    )


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message không được để trống")
    return detector.predict(request.message)


@app.get("/health")
def health():
    return {"status": "ok"}
