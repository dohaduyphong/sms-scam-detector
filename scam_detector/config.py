"""
Config
Đường dẫn model, tham số bộ ghép (ensemble) và threshold
"""
import os
from pathlib import Path

#   word_tfidf_vectorizer_full.joblib, logreg_model_full.joblib,
#   char_tfidf_vectorizer_full.joblib, svm_model_full.joblib
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
# Thư mục local hoặc HF Hub repo id của ViSoBERT đã fine-tune
VISOBERT_SOURCE = os.environ.get(
    "VISOBERT_SOURCE", str(MODEL_DIR / "visobert_full")
)

# setting inference ViSoBERT (giống lúc train / Ensemble_v2.ipynb)
VISOBERT_MAX_LEN = 256

# Bộ ghép stack_lr (Ensemble_v2.ipynb, outputs/ensemble_v2_config.json)
# Điểm thô s_i của từng model, theo thứ tự ENSEMBLE_MODELS:
#   word_lr    : logreg.decision_function(...)
#   char_svm   : logit(svm.predict_proba(...)[:, 1]), clip 1e-6
#   visobert   : logits[scam] - logits[ham]
# score = intercept + sum_i coef[i] * (s_i - scaler_mean[i]) / scaler_scale[i]
# scam nếu score >= ENSEMBLE_THRESHOLD
# Fit trên test_clean (522 tin, 109 scam), chính sách min_recall (recall >= 0.9)
ENSEMBLE_MODELS = ["word_lr", "char_svm", "visobert"]
ENSEMBLE_SCALER_MEAN = [-0.6854384490041919, -2.6533970678957357, -4.185691451404058]
ENSEMBLE_SCALER_SCALE = [1.6512267164198766, 3.8183008000240752, 5.880472176027398]
ENSEMBLE_COEF = [0.7341687086117116, 1.9483087939165036, 0.7795957699222847]
ENSEMBLE_INTERCEPT = -2.158993285248134
ENSEMBLE_THRESHOLD = 0.6844216153738869

# Origin được phép gọi API từ trình duyệt (CORS)
# - GitHub Pages của repo (frontend production)
# - localhost: docs/ chạy local bằng `python3 -m http.server 5500` hoặc Live Server (5501)
CORS_ORIGINS = [
    "https://dohaduyphong.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5501",
    "http://127.0.0.1:5501",
]
