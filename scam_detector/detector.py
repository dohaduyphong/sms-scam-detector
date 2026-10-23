"""
Nạp 4 mô hình và gộp điểm của chúng bằng stack_lr (Ensemble_v2.ipynb).
"""
import json
import math
from pathlib import Path
from typing import Callable, Dict, Optional, Union

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
    PHOBERT_MAX_LEN,
    PHOBERT_REVISION,
    PHOBERT_SOURCE,
    PHOBERT_TEMPERATURE,
    VISOBERT_MAX_LEN,
    VISOBERT_REVISION,
    VISOBERT_SOURCE,
    VISOBERT_TEMPERATURE,
)
from .preprocessing import prepare_text, prepare_text_phobert, prepare_text_visobert

def _load_label_mapping(source: str, revision: Optional[str] = None) -> dict:
    """
    from_pretrained() không tải label_mapping.json, nên đọc file này từ thư mục
    local hoặc tải riêng từ repo trên HF Hub.
    """
    local_path = Path(source) / "label_mapping.json"
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            return json.load(f)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=source, filename="label_mapping.json", revision=revision)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class _TransformerScorer:
    """Transformer phân loại hai lớp đã fine-tune, chỉ load một lần. Điểm trả về là logits[scam] - logits[ham]."""

    def __init__(
        self,
        source: str,
        revision: Optional[str],
        max_len: int,
        prepare: Callable[[str], str],
        device: torch.device,
    ):
        # Repo HF phải nạp đúng revision đã ghim; thư mục local thì không cần
        revision = None if Path(source).is_dir() else revision
        self.max_len, self.prepare, self.device = max_len, prepare, device
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            revision=revision,
            dtype=torch.float32,
        ).to(device)
        self.model.eval()

        label_map = _load_label_mapping(source, revision)
        # Vị trí của lớp có nhãn gốc là 1 (scam)
        self.scam_index = int(next(k for k, v in label_map.items() if int(v) == 1))
        self.ham_index = 1 - self.scam_index

    def score(self, raw_text: str) -> float:
        enc = self.tokenizer(
            self.prepare(raw_text),
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**enc).logits[0].float()
        return float(logits[self.scam_index] - logits[self.ham_index])


class ScamSMSDetector:
    def __init__(
        self,
        model_dir: Union[str, Path] = MODEL_DIR,
        visobert_source: Union[str, Path] = VISOBERT_SOURCE,
        phobert_source: Union[str, Path] = PHOBERT_SOURCE,
    ):
        model_dir = Path(model_dir)

        self.word_vectorizer = joblib.load(model_dir / f"word_tfidf_vectorizer.joblib")
        self.logreg_model = joblib.load(model_dir / f"logreg_model.joblib")

        self.char_vectorizer = joblib.load(model_dir / f"char_tfidf_vectorizer.joblib")
        self.svm_model = joblib.load(model_dir / f"svm_model.joblib")

        # decision_function và predict_proba[:, 1] đều coi lớp scam (1) là lớp dương
        assert list(self.logreg_model.classes_) == [0, 1]
        assert list(self.svm_model.classes_) == [0, 1]

        # Load hai transformer một lần lúc khởi động, không load lại ở mỗi request
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visobert = _TransformerScorer(
            str(visobert_source), VISOBERT_REVISION, VISOBERT_MAX_LEN, prepare_text_visobert, self.device
        )
        self.phobert = _TransformerScorer(
            str(phobert_source), PHOBERT_REVISION, PHOBERT_MAX_LEN, prepare_text_phobert, self.device
        )

        self.ens_mean = np.array(ENSEMBLE_SCALER_MEAN)
        self.ens_scale = np.array(ENSEMBLE_SCALER_SCALE)
        self.ens_coef = np.array(ENSEMBLE_COEF)

    def predict(self, raw_text: str) -> Dict:
        """
        Chạy cả bốn model trên một tin nhắn rồi gộp điểm thô của chúng
        bằng stacking logistic regression (stack_lr) trên z-score:
            score = intercept + sum_i coef[i] * (s_i - mean[i]) / scale[i]
        Tin là scam khi score >= ENSEMBLE_THRESHOLD. Tham số lấy từ Ensemble_v2.ipynb
        và được khai báo trong config.py.
        """

        # Tiền xử lý cho hai model TF-IDF; hai transformer tự tiền xử lý trong .score()
        clean_text = prepare_text(raw_text)

        # Logistic Regression trả về log-odds
        score_logreg = float(
            self.logreg_model.decision_function(
                self.word_vectorizer.transform([clean_text])
            )[0]
        )

        # SVM đã hiệu chỉnh (CalibratedClassifierCV) chỉ có xác suất, nên lấy logit của xác suất scam
        score_svm = _logit(float(
            self.svm_model.predict_proba(
                self.char_vectorizer.transform([clean_text])
            )[0, 1]
        ))

        score_visobert = self.visobert.score(raw_text)
        score_phobert = self.phobert.score(raw_text)

        # Thứ tự điểm phải khớp với ENSEMBLE_MODELS
        scores = np.array([score_logreg, score_svm, score_visobert, score_phobert])
        z = (scores - self.ens_mean) / self.ens_scale
        ensemble_score = float(ENSEMBLE_INTERCEPT + z @ self.ens_coef)
        is_scam = ensemble_score >= ENSEMBLE_THRESHOLD

        # proba_*: xác suất scam của từng model (sigmoid của điểm thô).
        # confidence_scam: xác suất của bộ stack_lr, cùng thang với threshold.
        return {
            "text": raw_text,
            "clean_text": clean_text,
            "proba_word_tfidf_logreg": _sigmoid(score_logreg),
            "proba_char_tfidf_svm": _sigmoid(score_svm),
            "proba_visobert": _sigmoid(score_visobert / VISOBERT_TEMPERATURE),
            "proba_phobert": _sigmoid(score_phobert / PHOBERT_TEMPERATURE),
            "ensemble_score": ensemble_score,
            "confidence_scam": _sigmoid(ensemble_score),
            "threshold": _sigmoid(ENSEMBLE_THRESHOLD),
            "label": "scam" if is_scam else "ham",
        }
