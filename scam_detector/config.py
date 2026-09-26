"""
Config
Điều chỉnh weight của từng mô hình, threshold
"""
from pathlib import Path

#   word_tfidf_vectorizer.joblib, logreg_model.joblib,
#   char_tfidf_vectorizer.joblib, svm_model.joblib
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Weight của từng model
MODEL_WEIGHTS = {
    "word_tfidf_logreg": 0.3,
    "char_tfidf_svm": 0.7,
}

# Scam threshold tổng thể
SCAM_THRESHOLD = 0.83