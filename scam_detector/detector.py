"""
Core phát hiện SMS Scam
- Load model đã huấn luyện, kết hợp kết quả của 3 mô hình
thành một kết quả phần trăm - độ tự tin của mô hình.
"""
from pathlib import Path
from typing import Dict, Union

import joblib

from .config import MODEL_DIR, MODEL_WEIGHTS, SCAM_THRESHOLD
from .preprocessing import preprocess_text


class ScamSMSDetector:
    def __init__(self, model_dir: Union[str, Path] = MODEL_DIR):
        model_dir = Path(model_dir)

        self.word_vectorizer = joblib.load(model_dir / "word_tfidf_vectorizer.joblib")
        self.logreg_model = joblib.load(model_dir / "logreg_model.joblib")

        self.char_vectorizer = joblib.load(model_dir / "char_tfidf_vectorizer.joblib")
        self.svm_model = joblib.load(model_dir / "svm_model.joblib")

    def predict(self, raw_text: str) -> Dict:
        """
        Chạy cả 2 mô hình trên một tin nhắn SMS do ng dùng nhập vào
        và kết hợp kết quả của chúng.

        Return một từ điển với kết quả của từng mô hình, điểm tổng.
        """
        
        clean_text = preprocess_text(raw_text)

        proba_logreg = self.logreg_model.predict_proba(
            self.word_vectorizer.transform([clean_text])
        )[0, 1]

        proba_svm = self.svm_model.predict_proba(
            self.char_vectorizer.transform([clean_text])
        )[0, 1]

        combined = (
            proba_logreg * MODEL_WEIGHTS["word_tfidf_logreg"]
            + proba_svm * MODEL_WEIGHTS["char_tfidf_svm"]
        )

        return {
            "text": raw_text,
            "clean_text": clean_text,
            "proba_word_tfidf_logreg": float(proba_logreg),
            "proba_char_tfidf_svm": float(proba_svm),
            "confidence_scam": float(combined),
            "label": "scam" if combined >= SCAM_THRESHOLD else "ham",
        }