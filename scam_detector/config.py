"""
Config
Đường dẫn model, tham số bộ ghép (ensemble) và threshold
"""
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
# Word/Char TF-IDF: word_tfidf_vectorizer.joblib, logreg_model.joblib,
#   char_tfidf_vectorizer.joblib, svm_model.joblib (trong MODEL_DIR)

# 2 transformer đã fine-tune: thư mục local hoặc HF Hub repo id.
# Mặc định: thư mục trong models/ nếu có (local), không thì tải từ HF (server).
# Revision (commit) cố định trên HF: tham số ensemble bên dưới chỉ đúng với đúng bản model này.
# Không dùng "main" -> tránh server nạp nhầm bản cũ/mới (cache, upload trước/sau deploy).
# Bỏ qua revision khi SOURCE là thư mục local. Đổi khi upload model mới + fit lại ensemble.
VISOBERT_HF_REPO = "dohaduyphong/visobert-scam-sms-vn"
VISOBERT_SOURCE = os.environ.get(
    "VISOBERT_SOURCE",
    str(MODEL_DIR / "visobert") if (MODEL_DIR / "visobert").is_dir() else VISOBERT_HF_REPO,
)
VISOBERT_REVISION = os.environ.get(
    "VISOBERT_REVISION", "ab6b90c8bbf53914c4d8a5c76237152f08d9c120"
)

PHOBERT_HF_REPO = "dohaduyphong/phobert-scam-sms-vn"
PHOBERT_SOURCE = os.environ.get(
    "PHOBERT_SOURCE",
    str(MODEL_DIR / "phobert_base_v2") if (MODEL_DIR / "phobert_base_v2").is_dir() else PHOBERT_HF_REPO,
)
PHOBERT_REVISION = os.environ.get("PHOBERT_REVISION", "d53d1a9bb93609233e92828ca6e2c6d48b502880")

# setting inference (giống lúc train / Ensemble_v2.ipynb)
# ViSoBERT: văn bản thô (NFC); PhoBERT: thêm tách từ pyvi (preprocessing.prepare_text_phobert)
VISOBERT_MAX_LEN = 256
PHOBERT_MAX_LEN = 256
# Temperature scaling CHỈ cho xác suất hiển thị (proba_visobert / proba_phobert): transformer rất tự tin
# (gần như luôn <1% hoặc >99%). sigmoid(score / T), T tối thiểu log-loss trên test_clean
# (ViSoBERT 0.593 -> 0.278, PhoBERT 0.317 -> 0.210).
# Không ảnh hưởng ensemble: ensemble dùng điểm thô đã chuẩn hoá z-score, chia T không đổi kết quả.
VISOBERT_TEMPERATURE = 3.18
PHOBERT_TEMPERATURE = 2.12

# Bộ ghép stack_lr (Ensemble_v2.ipynb, outputs/ensemble_v2_config.json)
# Điểm thô s_i của từng model, theo thứ tự ENSEMBLE_MODELS:
#   word_lr    : logreg.decision_function(...)
#   char_svm   : logit(svm.predict_proba(...)[:, 1]), clip 1e-6
#   visobert   : logits[scam] - logits[ham]
#   phobert    : logits[scam] - logits[ham]
# score = intercept + sum_i coef[i] * (s_i - scaler_mean[i]) / scaler_scale[i]
# scam nếu score >= ENSEMBLE_THRESHOLD
# Fit trên test_clean (897 tin, 250 scam), chính sách min_recall (recall >= 0.93),
# chọn theo F2 (Ensemble_v2.ipynb, chạy 06/10/2026).
# Cross-fitted: recall 0.928, precision 0.876, F2 0.917, FPR 0.051, PR-AUC 0.959
ENSEMBLE_MODELS = ["word_lr", "char_svm", "visobert", "phobert"]
ENSEMBLE_SCALER_MEAN = [-0.4184017196895849, -1.322913658225613, -3.28819343610485, -2.8636800591380167]
ENSEMBLE_SCALER_SCALE = [1.3443312190022245, 2.5040219412959686, 7.357235295636429, 5.69277014195548]
ENSEMBLE_COEF = [2.348104636747745, 0.35336019728563756, 0.2999415887974622, 1.3808207071729761]
ENSEMBLE_INTERCEPT = -1.748162373027225
ENSEMBLE_THRESHOLD = 0.555571785671755

# Origin được phép gọi API từ trình duyệt (CORS)
# - GitHub Pages của repo (frontend production)
# - localhost: docs/ chạy local bằng `python3 -m http.server 5500` hoặc Live Server (5501)
CORS_ORIGINS = [
    "https://dohaduyphong.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5501",
    "http://127.0.0.1:5501",
    "https://scam-detector.duyphong.info",
]
