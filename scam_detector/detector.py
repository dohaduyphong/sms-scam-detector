"""
Phần cốt lõi phát hiện scam
- Load các model
- Chạy các model trên SMS
- Gộp kết quả các model thành kết quả cuối cùng (stack_lr, Ensemble_v2.ipynb)
"""
import json
import math
from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np

import torch
torch.set_num_threads(2)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    ENSEMBLE_COEF,
    ENSEMBLE_INTERCEPT,
    ENSEMBLE_SCALER_MEAN,
    ENSEMBLE_SCALER_SCALE,
    ENSEMBLE_THRESHOLD,
    MODEL_DIR,
    VISOBERT_MAX_LEN,
    VISOBERT_SOURCE,
)
from .preprocessing import prepare_text, prepare_text_visobert

def _load_label_mapping(source: str) -> dict:
    """
    label_mapping.json isn't a standard transformers file, so
    from_pretrained() won't fetch it for us. Read it straight off disk
    if `source` is a local folder; otherwise treat `source` as a HF
    Hub repo id and download just that one file (cached after the
    first call, same as the model weights).
    """
    local_path = Path(source) / "label_mapping.json"
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            return json.load(f)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=source, filename="label_mapping.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ScamSMSDetector:
    def __init__(
        self,
        model_dir: Union[str, Path] = MODEL_DIR,
        visobert_source: Union[str, Path] = VISOBERT_SOURCE,
    ):
        model_dir = Path(model_dir)
        visobert_source = str(visobert_source)

        self.word_vectorizer = joblib.load(model_dir / f"word_tfidf_vectorizer.joblib")
        self.logreg_model = joblib.load(model_dir / f"logreg_model.joblib")

        self.char_vectorizer = joblib.load(model_dir / f"char_tfidf_vectorizer.joblib")
        self.svm_model = joblib.load(model_dir / f"svm_model.joblib")

        # decision_function / predict_proba[:, 1] giả định lớp scam (1) là lớp dương
        assert list(self.logreg_model.classes_) == [0, 1]
        assert list(self.svm_model.classes_) == [0, 1]

        # ViSoBERT -- load 1 lần từ đầu thay vì sau mỗi request
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visobert_tokenizer = AutoTokenizer.from_pretrained(visobert_source)
        self.visobert_model = AutoModelForSequenceClassification.from_pretrained(
            visobert_source,
            dtype=torch.float32,
        ).to(self.device)
        self.visobert_model.eval()

        label_map = _load_label_mapping(visobert_source)
        # index of the class whose original label value is 1 ("scam")
        self.visobert_scam_index = int(next(k for k, v in label_map.items() if int(v) == 1))
        self.visobert_ham_index = 1 - self.visobert_scam_index

        self.ens_mean = np.array(ENSEMBLE_SCALER_MEAN)
        self.ens_scale = np.array(ENSEMBLE_SCALER_SCALE)
        self.ens_coef = np.array(ENSEMBLE_COEF)

    def _score_visobert(self, text: str) -> float:
        # logits[scam] - logits[ham]
        enc = self.visobert_tokenizer(
            text,
            truncation=True,
            max_length=VISOBERT_MAX_LEN,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.visobert_model(**enc).logits[0].float()
        return float(logits[self.visobert_scam_index] - logits[self.visobert_ham_index])

    def predict(self, raw_text: str) -> Dict:
        """
        Chạy cả 3 model trên 1 tin nhắn sms rồi gộp điểm thô của chúng
        bằng stacking logistic regression (stack_lr) trên z-score:
            score = intercept + sum_i coef[i] * (s_i - mean[i]) / scale[i]
        scam nếu score >= ENSEMBLE_THRESHOLD
        - Tham số lấy từ Ensemble_v2.ipynb, chỉnh trong config.py
        """

        # tiền xử lý
        clean_text = prepare_text(raw_text)
        visobert_text = prepare_text_visobert(raw_text)

        # Logistic Regression: log-odds
        score_logreg = float(
            self.logreg_model.decision_function(
                self.word_vectorizer.transform([clean_text])
            )[0]
        )

        # SVM (CalibratedClassifierCV): logit của xác suất scam
        score_svm = _logit(float(
            self.svm_model.predict_proba(
                self.char_vectorizer.transform([clean_text])
            )[0, 1]
        ))

        # ViSoBERT
        score_visobert = self._score_visobert(visobert_text)

        # Kết hợp 3 kết quả
        scores = np.array([score_logreg, score_svm, score_visobert])
        z = (scores - self.ens_mean) / self.ens_scale
        ensemble_score = float(ENSEMBLE_INTERCEPT + z @ self.ens_coef)
        is_scam = ensemble_score >= ENSEMBLE_THRESHOLD

        # return một dictionary các kết quả
        # proba_* : xác suất scam của từng model (sigmoid của điểm thô)
        # confidence_scam : xác suất của bộ stack_lr; threshold cùng thang
        return {
            "text": raw_text,
            "clean_text": clean_text,
            "proba_word_tfidf_logreg": _sigmoid(score_logreg),
            "proba_char_tfidf_svm": _sigmoid(score_svm),
            "proba_visobert": _sigmoid(score_visobert),
            "ensemble_score": ensemble_score,
            "confidence_scam": _sigmoid(ensemble_score),
            "threshold": _sigmoid(ENSEMBLE_THRESHOLD),
            "label": "scam" if is_scam else "ham",
        }
