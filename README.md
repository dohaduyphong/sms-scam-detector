# Vietnamese SMS Scam Detector

An AI-powered system for detecting scam SMS messages in Vietnamese.

The project analyzes the content of an SMS and predicts whether it is **scam** or **ham (legitimate)** using an ensemble of three machine-learning models:

* **Word TF-IDF + Logistic Regression**
* **Character TF-IDF + Linear SVM**
* **PhoBERT**

The system is designed specifically for Vietnamese SMS and includes text normalization, personal-information masking, model inference, ensemble decision-making, and a web API.

## Purpose

This repository contains the implementation of a Vietnamese SMS scam detection system, from text preprocessing and trained model inference to ensemble classification and web API deployment.

The main focus is on combining **traditional NLP models based on TF-IDF** with a **Vietnamese transformer model (PhoBERT)** to analyze scam-related patterns from different representations of the same SMS.

## How It Works

For each SMS, the system follows this pipeline:

```text
SMS message
     │
     ▼
PII & URL detection
     │
     ├───────────────┐
     ▼               ▼
TF-IDF pipeline   PhoBERT pipeline
     │               │
     ├───────┐   ┌───┘
     ▼       ▼   ▼
  Logistic  SVM  PhoBERT
     │       │     │
     └───────┴─────┘
             │
             ▼
    Weighted Soft Voting
             │
             ▼
      Safety Override
             │
             ▼
       Scam / Ham
```

### 1. Text preprocessing

The system first identifies and normalizes information that can vary between messages.

It detects and replaces:

* Vietnamese phone numbers
* Bank account numbers
* Dates
* Times
* Monetary amounts
* Other numerical identifiers
* URLs

These values are converted into standardized tokens such as `[PHONE]`, `[BANK_ACC]`, `[DATE]`, `[TIME]`, `[MONEY]`, `[NUMBER]`, and `[LINK]`.

This allows the models to focus more on the linguistic patterns of a message instead of memorizing individual phone numbers, account numbers, URLs, or amounts.

The TF-IDF models use a more aggressive normalization pipeline, including lowercasing, URL replacement, punctuation removal, and whitespace normalization.

PhoBERT uses a separate preprocessing pipeline because it preserves casing and punctuation. The text is tagged for PII and links and then word-segmented using `PyVi`.

## Models

### Word TF-IDF + Logistic Regression

The first model represents the SMS using word-level TF-IDF features and classifies the message with Logistic Regression.

It captures scam-related patterns at the word level and provides a probability for the scam class.

### Character TF-IDF + Linear SVM

The second model uses character-level TF-IDF features with a Linear SVM.

Character-level features are useful for Vietnamese SMS because they can capture variations in spelling, abbreviations, obfuscation, and messages containing unusual formatting.

This model currently has the largest weight in the final ensemble.

### PhoBERT

The third model is a fine-tuned PhoBERT sequence-classification model for Vietnamese text.

Unlike the TF-IDF models, PhoBERT operates on word-segmented Vietnamese text and uses a transformer-based representation to capture contextual relationships within the SMS.

The model can be loaded from a local directory or from a Hugging Face Hub repository.

## Ensemble Decision

The three models independently produce a probability that the SMS is a scam.

The final confidence is calculated using a weighted combination:

```text
P(scam) =
    0.10 × P(Logistic Regression)
  + 0.75 × P(SVM)
  + 0.15 × P(PhoBERT)
```

The current weights are defined in `scam_detector/config.py`.

The resulting score is compared against a scam threshold of `0.70`.

The system also contains an **override mechanism**. If any individual model produces a scam probability of at least `0.75`, the message is classified as scam regardless of the ensemble score.

This prevents a highly confident prediction from being completely canceled out by the other models.

## Output

For every analyzed SMS, the detector returns:

```json
{
  "text": "...",
  "clean_text": "...",
  "proba_word_tfidf_logreg": 0.12,
  "proba_char_tfidf_svm": 0.91,
  "proba_phobert": 0.87,
  "confidence_scam": 0.88,
  "override_triggered": false,
  "label": "scam"
}
```

The output contains both the individual model probabilities and the final ensemble result.

## Project Structure

```text
sms-scam-detector/
│
├── docs/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/
│   ├── word_tfidf_vectorizer_full.joblib
│   ├── logreg_model_full.joblib
│   ├── char_tfidf_vectorizer_full.joblib
│   ├── svm_model_full.joblib
│   └── phobert_full/
│
├── scam_detector/
│   ├── __init__.py
│   ├── config.py
│   ├── detector.py
│   └── preprocessing.py
│
├── api.py
├── main.py
├── Dockerfile
├── requirements.txt
└── push_phobert_to_hub.py
```

The main detection logic is implemented in `scam_detector/detector.py`. It loads all trained models, performs inference, combines their predictions, and produces the final classification.

`scam_detector/preprocessing.py` contains the Vietnamese text preprocessing and PII/link detection pipeline.

`scam_detector/config.py` contains model weights and classification thresholds.

## Command-Line Interface

The detector can be used directly from the command line.

```bash
python3 main.py "Chuc mung ban da trung thuong, lien he STK de nhan qua"
```

It also supports an interactive mode:

```bash
python3 main.py
```

The CLI displays the prediction probability from each model, the final scam confidence, and the final label.

## REST API

The project provides a FastAPI backend.

Start the API with:

```bash
uvicorn api:app --reload --port 8000
```

The main endpoint is:

```text
POST /predict
```

with a request body such as:

```json
{
  "message": "Chuc mung ban trung thuong..."
}
```

The API returns the prediction generated by the ensemble detector.

A health-check endpoint is also available:

```text
GET /health
```

which returns:

```json
{
  "status": "ok"
}
```

The API is configured with CORS support for the project's web frontend.

## Web Interface

The `docs/` directory contains the static web frontend.

The frontend communicates with the FastAPI `/predict` endpoint and provides an interface for entering an SMS and displaying the model's prediction.

For local development, the frontend can be served with:

```bash
cd docs
python3 -m http.server 5500
```

while the FastAPI backend runs separately on port `8000`.

## Deployment

The repository includes a `Dockerfile` configured for container-based deployment.

The container uses Python 3.11 and starts the FastAPI application on port `7860`, making it suitable for platforms that expect a Docker-based web service on that port.

The project can therefore be structured as:

```text
Web frontend
     │
     │ HTTP POST /predict
     ▼
FastAPI backend
     │
     ▼
ScamSMSDetector
     │
     ├── TF-IDF + Logistic Regression
     ├── TF-IDF + Linear SVM
     └── PhoBERT
     │
     ▼
Ensemble prediction
     │
     ▼
Scam / Ham
```

## Technologies

* Python
* FastAPI
* scikit-learn
* PyTorch
* Hugging Face Transformers
* PhoBERT
* Joblib
* PyVi
* HTML / CSS / JavaScript
* Docker

The required Python packages and versions are defined in `requirements.txt`.
