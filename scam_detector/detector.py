"""
Phần cốt lõi phát hiện scam
- Load các model
- Chạy các model trên SMS
- Gộp kết quả các model thành kết quả cuối cùng
"""
import json
from pathlib import Path
from typing import Dict, Union

import joblib

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    MODEL_DIR,
    MODEL_WEIGHTS,
    OVERRIDE_THRESHOLD,
    PHOBERT_SOURCE,
    PHOBERT_MAX_LEN,
    SCAM_THRESHOLD,
)
from .preprocessing import prepare_text, prepare_text_phobert

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


class ScamSMSDetector:
    def __init__(
        self,
        model_dir: Union[str, Path] = MODEL_DIR,
        phobert_source: Union[str, Path] = PHOBERT_SOURCE,
    ):
        model_dir = Path(model_dir)
        """
        self.word_vectorizer = joblib.load(model_dir / "word_tfidf_vectorizer.joblib")
        self.logreg_model = joblib.load(model_dir / "logreg_model.joblib")

        self.char_vectorizer = joblib.load(model_dir / "char_tfidf_vectorizer.joblib")
        self.svm_model = joblib.load(model_dir / "svm_model.joblib")
        """
        self.word_vectorizer = joblib.load(model_dir / "word_tfidf_vectorizer_full.joblib")
        self.logreg_model = joblib.load(model_dir / "logreg_model_full.joblib")
        
        self.char_vectorizer = joblib.load(model_dir / "char_tfidf_vectorizer_full.joblib")
        self.svm_model = joblib.load(model_dir / "svm_model_full.joblib")


        self.logreg_scam_index = list(self.logreg_model.classes_).index(1)
        self.svm_scam_index = list(self.svm_model.classes_).index(1)

        # PhoBERT -- load 1 lần từ đầu thay vì sau mỗi request
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phobert_tokenizer = AutoTokenizer.from_pretrained(
            phobert_source, use_fast=False
        )
        self.phobert_model = AutoModelForSequenceClassification.from_pretrained(
            phobert_source
        ).to(self.device)
        self.phobert_model.eval()

        label_map = _load_label_mapping(phobert_source)
        # index of the class whose original label value is 1 ("scam")
        self.phobert_scam_index = [int(k) for k, v in label_map.items() if v == 1][0]

    def _predict_phobert_proba(self, segmented_text: str) -> float:
        with torch.no_grad():
            enc = self.phobert_tokenizer(
                segmented_text,
                padding=True,
                truncation=True,
                max_length=PHOBERT_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)
            logits = self.phobert_model(**enc).logits
            probs = F.softmax(logits, dim=1)
        return float(probs[0, self.phobert_scam_index])

    def predict(self, raw_text: str) -> Dict:
        """
        Chạy cả 3 model trên 1 tin nhắn sms sau đó kết hợp
        kết quả của chúng thành 1 chỉ số tự tin duy nhất.
        Kết hợp với safety override trong trường hợp 1 mô hình rất tự tin
        nhưng bị outvote bởi các mô hình khác
        - Chỉnh threshold trong config.py
        """

        # tiền xử lý
        clean_text = prepare_text(raw_text)
        phobert_text = prepare_text_phobert(raw_text)

        # check tiền xử lý
        print(clean_text)
        print(phobert_text)

        # Logistic Regression
        proba_logreg = float(
            self.logreg_model.predict_proba(
                self.word_vectorizer.transform([clean_text])
            )[0, self.logreg_scam_index]
        )

        # SVM
        proba_svm = float(
            self.svm_model.predict_proba(
                self.char_vectorizer.transform([clean_text])
            )[0, self.svm_scam_index]
        )

        # PhoBERT
        proba_phobert = self._predict_phobert_proba(phobert_text)

        # Kết hợp 3 kết quả
        combined = (
            proba_logreg * MODEL_WEIGHTS["word_tfidf_logreg"]
            + proba_svm * MODEL_WEIGHTS["char_tfidf_svm"]
            + proba_phobert * MODEL_WEIGHTS["phobert"]
        )

        # Override
        override_triggered = (
            proba_logreg >= OVERRIDE_THRESHOLD
            or proba_svm >= OVERRIDE_THRESHOLD
            or proba_phobert >= OVERRIDE_THRESHOLD
        )
        is_scam = (combined >= SCAM_THRESHOLD) or override_triggered

        # return một dictionary các kết quả
        return {
            "text": raw_text,
            "clean_text": clean_text,
            "proba_word_tfidf_logreg": float(proba_logreg),
            "proba_char_tfidf_svm": float(proba_svm),
            "proba_phobert": float(proba_phobert),
            "confidence_scam": float(combined),
            "override_triggered": override_triggered,
            "label": "scam" if is_scam else "ham",
        }