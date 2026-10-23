"""
Đường dẫn model, tham số của bộ kết hợp (ensemble) và ngưỡng phân loại.
"""
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
# Hai model TF-IDF nằm trong MODEL_DIR: word_tfidf_vectorizer.joblib, logreg_model.joblib,
#   char_tfidf_vectorizer.joblib, svm_model.joblib.

# Hai transformer đã fine-tune được lấy từ thư mục local hoặc từ repo trên HF Hub.
# Mặc định dùng thư mục trong models/ nếu có (máy local), nếu không thì tải từ HF (máy chủ).
# Revision trên HF được ghim vì tham số ensemble chỉ khớp với đúng bản model này; dùng thư mục local thì bỏ qua.
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

# Thiết lập lúc suy luận, giống lúc train (Ensemble_v2.ipynb).
# ViSoBERT nhận văn bản thô đã chuẩn hoá NFC; PhoBERT còn được tách từ bằng pyvi (preprocessing.prepare_text_phobert).
VISOBERT_MAX_LEN = 256
PHOBERT_MAX_LEN = 256
# Temperature chỉ dùng để tính xác suất hiển thị, sigmoid(score / T), với T được fit trên test_clean.
# Ensemble dùng z-score của điểm thô nên không bị ảnh hưởng.
VISOBERT_TEMPERATURE = 3.18
PHOBERT_TEMPERATURE = 2.12

# Bộ kết hợp stack_lr (Ensemble_v2.ipynb, outputs/ensemble_v2_config.json).
# Điểm thô s_i của từng model, theo thứ tự ENSEMBLE_MODELS:
#   word_lr    : logreg.decision_function(...)
#   char_svm   : logit(svm.predict_proba(...)[:, 1]), clip 1e-6
#   visobert   : logits[scam] - logits[ham]
#   phobert    : logits[scam] - logits[ham]
# score = intercept + sum_i coef[i] * (s_i - scaler_mean[i]) / scaler_scale[i]
# Tin là scam khi score >= ENSEMBLE_THRESHOLD.
# Fit trên test_clean (897 tin, 250 tin scam) theo chính sách min_recall (recall >= 0.93),
# chọn phương án theo F2 (Ensemble_v2.ipynb, chạy ngày 06/10/2026).
# Kết quả cross-fitted: recall 0.928, precision 0.876, F2 0.917, FPR 0.051, PR-AUC 0.959
ENSEMBLE_MODELS = ["word_lr", "char_svm", "visobert", "phobert"]
ENSEMBLE_SCALER_MEAN = [-0.4184017196895849, -1.322913658225613, -3.28819343610485, -2.8636800591380167]
ENSEMBLE_SCALER_SCALE = [1.3443312190022245, 2.5040219412959686, 7.357235295636429, 5.69277014195548]
ENSEMBLE_COEF = [2.348104636747745, 0.35336019728563756, 0.2999415887974622, 1.3808207071729761]
ENSEMBLE_INTERCEPT = -1.748162373027225
ENSEMBLE_THRESHOLD = 0.555571785671755

# Các origin được phép gọi API từ trình duyệt (CORS):
# - GitHub Pages của repo (frontend chính thức)
# - localhost, khi chạy docs/ trên máy bằng `python3 -m http.server 5500` hoặc Live Server (cổng 5501)
CORS_ORIGINS = [
    "https://dohaduyphong.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5501",
    "http://127.0.0.1:5501",
    "https://scam-detector.duyphong.info",
]
