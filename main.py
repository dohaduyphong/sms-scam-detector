"""
Main
"""
import sys

from scam_detector import ScamSMSDetector


def print_result(result: dict) -> None:
    print(f"\nMessage : {result['text']}")
    print(f"Word TF-IDF + LogReg : {result['proba_word_tfidf_logreg']:.3f}")
    print(f"Char TF-IDF + SVM    : {result['proba_char_tfidf_svm']:.3f}")
    print(f"ViSoBERT             : {result['proba_visobert']:.3f}")
    print(f"PhoBERT              : {result['proba_phobert']:.3f}")
    print(f"Confidence (scam)    : {result['confidence_scam']:.1%}")
    print(f"Label                : {result['label'].upper()}")


def main() -> None:
    detector = ScamSMSDetector()

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        print_result(detector.predict(message))
        return

    print("Scam SMS Detector — nhap tin nhan de kiem tra (Ctrl+C de thoat)\n")
    while True:
        try:
            message = input("SMS > ").strip()
            if not message:
                continue
            print_result(detector.predict(message))
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()