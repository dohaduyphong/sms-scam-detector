"""
Config
Đường dẫn model, tham số bộ ghép (ensemble) và threshold
"""
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
# Bộ model TF-IDF đang dùng: word_tfidf_vectorizer{MODEL_SUFFIX}.joblib, logreg_model{MODEL_SUFFIX}.joblib,
#   char_tfidf_vectorizer{MODEL_SUFFIX}.joblib, svm_model{MODEL_SUFFIX}.joblib
# Thư mục local hoặc HF Hub repo id của ViSoBERT đã fine-tune.
# Mặc định: models/visobert nếu có (local), không thì tải từ HF (server chưa set biến môi trường)
VISOBERT_HF_REPO = "dohaduyphong/visobert-scam-sms-vn"
VISOBERT_SOURCE = os.environ.get(
    "VISOBERT_SOURCE",
    str(MODEL_DIR / "visobert") if (MODEL_DIR / "visobert").is_dir() else VISOBERT_HF_REPO,
)
# Revision (commit) cố định trên HF: tham số ensemble bên dưới chỉ đúng với đúng bản model này.
# Không dùng "main" -> tránh server nạp nhầm bản cũ/mới (cache, upload trước/sau deploy).
# Bỏ qua khi VISOBERT_SOURCE là thư mục local. Đổi khi upload model mới + fit lại ensemble.
VISOBERT_REVISION = os.environ.get(
    "VISOBERT_REVISION", "ab6b90c8bbf53914c4d8a5c76237152f08d9c120"
)

# setting inference ViSoBERT (giống lúc train / Ensemble_v2.ipynb)
VISOBERT_MAX_LEN = 256
# Temperature scaling CHỈ cho xác suất hiển thị (proba_visobert): ViSoBERT rất tự tin
# (94% dự đoán <1% hoặc >99%). sigmoid(score / T) với T fit trên test_clean (log-loss 0.593 -> 0.278).
# Không ảnh hưởng ensemble: ensemble dùng điểm thô đã chuẩn hoá z-score, chia T không đổi kết quả.
VISOBERT_TEMPERATURE = 3.18

# Bộ ghép stack_lr (Ensemble_v2.ipynb, outputs/ensemble_v2_config.json)
# Điểm thô s_i của từng model, theo thứ tự ENSEMBLE_MODELS:
#   word_lr    : logreg.decision_function(...)
#   char_svm   : logit(svm.predict_proba(...)[:, 1]), clip 1e-6
#   visobert   : logits[scam] - logits[ham]
# score = intercept + sum_i coef[i] * (s_i - scaler_mean[i]) / scaler_scale[i]
# scam nếu score >= ENSEMBLE_THRESHOLD
# Fit trên test_clean (897 tin, 250 scam), chính sách min_recall (recall >= 0.93),
# chọn theo F2 (Ensemble_v2.ipynb, chạy 05/10/2026). Cross-fitted: recall 0.932, FPR 0.071, F2 0.911
ENSEMBLE_MODELS = ["word_lr", "char_svm", "visobert"]
ENSEMBLE_SCALER_MEAN = [-0.4184017196895849, -1.322913658225613, -3.28819343610485]
ENSEMBLE_SCALER_SCALE = [1.3443312190022245, 2.5040219412959686, 7.357235295636429]
ENSEMBLE_COEF = [2.3261505196760788, 0.8584699830698975, 1.049752457275563]
ENSEMBLE_INTERCEPT = -1.7571297591911594
ENSEMBLE_THRESHOLD = -0.03048477436497743

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
