# Scam SMS Detector (Python)

Python package wrapping the two trained models (Word TF-IDF +
Logistic Regression, Char TF-IDF + Linear SVM) exported from Colab.

## Setup

1. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

2. CLI test:
   ```bash
   python3 main.py "Chuc mung ban da trung thuong..."
   ```
   or 
   ```bash
   python3 main.py
   ```
   for interactive mode.

## Running locally (backend + frontend)

Two servers, in two terminals, both from inside `scam_sms_detector/`:

```bash
# Terminal 1 — backend API on port 8000
python3 -m uvicorn api:app --reload --port 8000

# Terminal 2 — static frontend on port 5500
cd docs && python3 -m http.server 5500
```

Open `http://localhost:5500`. `docs/script.js` already points at
`http://localhost:8000/predict`, so no config needed for local dev.

Quick backend-only check (no browser needed):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Chuc mung ban trung thuong, lien he STK ... de nhan qua"}'
```