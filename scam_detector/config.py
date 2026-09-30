"""
Config
Điều chỉnh weight của từng model, threshold
"""
import os
from pathlib import Path

#   word_tfidf_vectorizer.joblib, logreg_model.joblib,
#   char_tfidf_vectorizer.joblib, svm_model.joblib
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
# PHOBERT_DIR = MODEL_DIR / "phobert"
PHOBERT_SOURCE = os.environ.get(
    "PHOBERT_SOURCE", str(MODEL_DIR / "phobert_full")
)
# Weight của từng model
MODEL_WEIGHTS = {
    "word_tfidf_logreg": 0.04,
    "char_tfidf_svm": 0.62,
    "phobert": 0.34,
}

# Scam threshold tổng thể
SCAM_THRESHOLD = 0.4

# Giá trị ngưỡng scam
# Nếu có 1 mô hình có confidence score > OVERRIDE_THRESHOLD, bỏ qua các mô hình khác và mặc định SMS đó là scam
OVERRIDE_THRESHOLD = 0.95

# setting inference PhoBERT
PHOBERT_MAX_LEN = 64