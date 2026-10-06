# Vietnamese SMS Scam Detector

A system that classifies Vietnamese SMS messages as **scam** or **ham (legitimate)**.

Four models read the same message in different ways, and a stacking ensemble combines what they see:

| Model | What it looks at |
|---|---|
| **Word TF-IDF + Logistic Regression** | Words and two-word phrases ("chuyển khoản", "khóa tài khoản") |
| **Character TF-IDF + Linear SVM** | 3–5 character fragments. These still match when a word is misspelled, unaccented or deliberately obfuscated (`Gja'`, `1OO%`, `T-D-T-C`) |
| **PhoBERT** (fine-tuned transformer, word-segmented input) | The meaning of the whole sentence in standard Vietnamese. It works on short messages and messages without obvious keywords |
| **ViSoBERT** (fine-tuned transformer, social-media pre-training) | The meaning of the whole sentence, pre-trained on informal text. Weaker on its own, but it makes different mistakes from PhoBERT and keeps the ensemble stable on unaccented and chat-style messages |

The ensemble catches **93% of scam messages** and flags **5% of legitimate messages** by mistake. These numbers come from a held-out test set that includes deliberately hard cases ([Results](#results)).

---

## How a message is classified

```text
raw SMS
   │
   │  1. tag_links → tag_pii        replace links and personal data with placeholder tokens
   ▼
"Chuyen [MONEY] vao STK [BANK_ACC], lien he [PHONE] hoac truy cap [LINK]"
   │
   ├─► lowercase, strip punctuation ─► Word TF-IDF (1–2 grams) ─► Logistic Regression ─► s_word
   ├─► lowercase, strip punctuation ─► Char TF-IDF (3–5 grams) ─► Linear SVM          ─► s_char
   ├─► Unicode NFC, keep case/punct ─► ViSoBERT tokenizer      ─► ViSoBERT            ─► s_viso
   └─► NFC + word segmentation      ─► PhoBERT tokenizer       ─► PhoBERT             ─► s_pho
                                                    │
                     2. standardise each score:  z_i = (s_i − mean_i) / std_i
                     3. stacking:               score = b + w₁·z_word + w₂·z_char + w₃·z_viso + w₄·z_pho
                                                    │
                                     scam  if score ≥ threshold, else ham
```

### 1. Replacing personal data and links (`scam_detector/preprocessing.py`)

Phone numbers, account numbers, amounts and URLs differ from message to message. They identify a person but say nothing about whether the message is a scam. If they stayed in the text, the models would memorise specific numbers instead of learning the wording of scams. They are therefore replaced with placeholder tokens. The token names and the rules for which numbers are replaced follow the conventions of the training data:

| Token | Detected from |
|---|---|
| `[LINK]` | `http(s)://…`, `www.…`, bare domains such as `abc.com.vn`. Links are tagged first, so digits inside a URL are not mistaken for numbers |
| `[NAME]` | Vietnamese person names: capitalised words after a context word ("Người nhận:", "Quý khách", "Ông/Bà:", "anh", "chị"…) or starting with a common surname (Nguyễn, Trần, Phạm…). Words that are not valid Vietnamese syllables (brands such as Zalo or Viettel, and abbreviations) are rejected |
| `[PHONE]` | Vietnamese mobile and landline numbers (`0[235789]…`, `+84…`) and 1800/1900 hotlines |
| `[BANK_ACC]` | 6–19 digits next to a bank keyword ("STK", "TK", "tài khoản") or a bank name (Vietcombank, MB, TPBank…) |
| `[DATE]` | `20/09/2026`, `31/08`, `ngày 20 tháng 9` |
| `[TIME]` | `14:30`, `14h30`, `22 giờ` |
| `[MONEY]` | `500k`, `1.5tr`, `2 triệu`, `500.000đ`, `1.000.000` |
| `[NUMBER]` | Any other standalone number with ≥ 4 digits (OTP codes, order IDs, SMS short codes) |

Short numbers are deliberately left as they are, for example "giảm **60%**", "**3**GB", "tặng **1** phần quà". The dataset keeps them, and they carry meaning.

The two TF-IDF models then lowercase the text and strip punctuation. The placeholder tokens are protected while this happens, so `[MONEY]` stays a single token. ViSoBERT gets the text with case and punctuation kept, only Unicode-normalised (NFC). PhoBERT gets the same text after word segmentation with `pyvi` ("trúng thưởng" → "trúng_thưởng"), the form it was pre-trained on; placeholder tokens are kept intact during segmentation.

### 2–3. Combining the four models (`scam_detector/detector.py`)

The four models produce scores on different scales, so each one is first converted to log-odds:

- Logistic Regression: `decision_function` (log-odds)
- Linear SVM: `logit(P(scam))` from the calibrated SVM
- ViSoBERT, PhoBERT: `logit_scam − logit_ham`

Each score is then standardised to a z-score. A logistic-regression meta-model (**stacking**) combines the four z-scores. Its coefficients come from the ensemble study described below:

| | Word LR | Char SVM | ViSoBERT | PhoBERT |
|---|---|---|---|---|
| Stacking coefficient | 2.35 | 0.35 | 0.30 | 1.38 |

The coefficients, scaler statistics and decision threshold are stored in `scam_detector/config.py`.

---

## Training pipeline

```text
Vietnamese SMS dataset (+ hard cases added by hand)
   │  retag: re-tag PII so training and inference use exactly the same tokens
   ▼
train.csv (2,978)             test.csv (1,127)
   │                             │  remove near-duplicates and messages that also appear in train
   │                             ▼
   │                          test_clean (897)  ── the only data no model has seen
   ├─► Word TF-IDF + LR          │
   ├─► Char TF-IDF + SVM         │
   ├─► ViSoBERT fine-tune        │
   └─► PhoBERT fine-tune         │
            │                    │
            └──── score ─────────┤
                                 ▼
             ensemble study: compare ~10 ways to combine, cross-fitted on test_clean
                                 ▼
             stacking coefficients + threshold  →  scam_detector/config.py
```

### Data

- **Source.** *Vietnamese SMS Dataset with Quality Assurance* (Tran et al., 2026, CC BY 4.0). It contains 2,991 real messages. Personal data was already anonymised in the original release.
- **Hard cases.** These were added by hand to both splits:
  - informal personal messages that mention money transfers or links ("c ơi e vừa ck tiền hàng…", "check stk momo xem nổi chưa");
  - scams that impersonate government agencies and banks in formal language (traffic fines, tax refunds, "Smart OTP upgrade").


  |  | Messages | Ham | Scam |
  |---|---|---|---|
  | train | 2,978 | 2,159 | 819 |
  | test | 1,127 | 762 | 365 |

- **Retagging.** The original release merges phone numbers and account numbers into `[NUMBER]`. A retagging script uses the surrounding words to put them back as `[PHONE]` or `[BANK_ACC]`: "LH / CSKH / Hotline / Zalo `[NUMBER]`" becomes a phone number, and "STK / TK `[NUMBER]`" becomes an account number. The script then runs the same `tag_pii` that the detector uses at prediction time. The models are therefore trained on exactly the token distribution they will see in production.
- **Leakage control.** Messages are normalised: lowercased, accents removed, `đ` written as `d`. They are compared by word-set Jaccard similarity, and two messages with similarity ≥ 0.95 count as duplicates. Duplicates inside each split are collapsed. Test messages that are near-duplicates of a training message are removed. The result is **`test_clean`**: 897 messages, 250 of them scam.

### Model 1 — Word TF-IDF + Logistic Regression

- `TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000, min_df=2)`
- `LogisticRegression(class_weight='balanced', max_iter=1000)`

Each message activates about 54 features on average, and every one of them is a meaningful phrase. Logistic Regression gives probabilities directly, and its coefficients can be read per word.

### Model 2 — Character TF-IDF + Linear SVM

- `TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000, min_df=2)`
- `CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=5000), cv=5)`

Character n-grams overlap heavily. The word "khoan" alone produces `kho`, `hoa`, `oan`, `khoa`, `hoan`… so each message activates about 250 features that are strongly correlated with each other. A max-margin SVM handles this kind of high-dimensional, redundant input better than Logistic Regression: in a controlled comparison on `test_clean`, SVM reached PR-AUC 0.914 and LR 0.890. The SVM is wrapped in a calibration step to turn its output into a probability.

### Models 3–4 — ViSoBERT and PhoBERT

- `uitnlp/visobert` is an XLM-RoBERTa model pre-trained on Vietnamese social-media text. It reads raw text.
- `vinai/phobert-base-v2` is a RoBERTa model pre-trained on standard Vietnamese (news, Wikipedia) that was word-segmented first, so its input is segmented with `pyvi` in the same way.

Both are fine-tuned for binary classification with the same recipe. The choices below were made after studying where the earlier version failed:

| Choice | Why |
|---|---|
| **Placeholder tokens added as special tokens.** Their embeddings start from a related word: `[MONEY]` from "tiền", `[PHONE]` from "điện thoại" | The original tokenizer split `[MONEY]` into five meaningless pieces (`▁[ · M · ONE · Y · ]`), which hid the strongest scam signal in the data |
| **Best checkpoint chosen by scam-class PR-AUC** on `logit_scam − logit_ham` | That is exactly the score the ensemble uses. The earlier weighted-F1 criterion was dominated by the ham class, which makes up 73% of the data |
| **Validation set = 1/6 of train**, split by groups of messages from the same template (word-set Jaccard ≥ 0.6, `StratifiedGroupKFold`) | About 150 scam messages, enough to pick a checkpoint reliably. With a 0.95 grouping, 43% of validation messages had a sibling (Jaccard ≥ 0.6) in the training part, against 10% of `test_clean`, so validation looked far easier than the test set (PR-AUC ≈ 0.99) |
| **Unaccented copies** of the training messages that have accents (train part only, after the split) | More than half of real SMS are written without diacritics, and every model was weaker on them |
| lr 3e-5, batch 16, up to 8 epochs, early stopping after 2 epochs without improvement, warmup 10%, weight decay 0.01, max length 256, fp16 on GPU | Converges in a few epochs on about 3–4k examples |

After these changes, ViSoBERT's recall on the hard test cases (at its default threshold) rose from 0.60 to 0.90.

**Why PhoBERT was added.** Five transformers were trained with this recipe on Kaggle (`Model_Train_Transformers_Compare.ipynb`) and compared with 19 classical and light neural models on `test_clean` (`Model_Comparison.ipynb`). PhoBERT was the strongest transformer in each of three training runs (PR-AUC 0.940–0.954; XLM-R, mBERT and DistilmBERT 0.85–0.91; ViSoBERT 0.90). Replacing ViSoBERT with PhoBERT lowered the ensemble's false-positive rate significantly, but lost recall on unaccented and chat-style messages. Keeping both gave the best balance (see [Results](#results)).

### Ensemble study

All four models have already seen `train.csv`. Their scores on it would be overly optimistic, so the way to combine them is learned on **`test_clean`** instead. Cross-fitting keeps that estimate honest:

1. `test_clean` is split into 5 folds, grouped by near-duplicates.
2. For each fold, the combination method and its threshold are fitted on the other 4 folds and then used to predict that fold. Every message is therefore predicted by a model that never saw it.
3. **Threshold policy (catch scams first).** Among thresholds that keep recall ≥ 0.93, choose the one with the highest precision.
4. **Selection rule.** Keep the methods that reach the recall target, then pick the one with the highest **F2**. F2 weights recall about four times as much as precision. Ties are broken by PR-AUC.

Eleven options were compared:
- each model on its own;
- voting (1, 2 or 3 of 4 models);
- average of z-scores, average of percentile ranks, maximum z-score;
- a grid search over weights;
- **stacking with logistic regression**, which was selected.

A drop-one study then checks that every model actually contributes.

---

## Results

Cross-fitted results on `test_clean` (897 messages, 250 scam):

| Method | Recall | Precision | F2 | False-positive rate | PR-AUC |
|---|---|---|---|---|---|
| **Stacking ensemble (4 models)** | **0.928** | **0.876** | **0.917** | **0.051** | **0.959** |
| Average of z-scores | 0.928 | 0.869 | 0.916 | 0.054 | 0.954 |
| Previous stacking ensemble (Word LR + Char SVM + ViSoBERT) | 0.932 | 0.835 | 0.911 | 0.071 | 0.943 |
| PhoBERT only | 0.920 | 0.810 | 0.896 | 0.084 | 0.954 |
| Char TF-IDF + SVM only | 0.932 | 0.766 | 0.893 | 0.110 | 0.908 |
| ViSoBERT only | 0.928 | 0.758 | 0.888 | 0.114 | 0.900 |
| Word TF-IDF + LR only | 0.924 | 0.760 | 0.886 | 0.113 | 0.932 |

The false-positive rate is the share of legitimate messages wrongly flagged as scam. 95% bootstrap confidence intervals for the ensemble: F2 [0.888, 0.943], false-positive rate [0.035, 0.069]. Compared with the previous three-model ensemble, the false-positive rate drops by 2 points (paired bootstrap 95% CI [−3.3, −0.6]) at the same recall.

**Drop-one study:**

| Removed | F2 | False-positive rate |
|---|---|---|
| — (all four) | 0.917 | 0.051 |
| Word LR | 0.913 | 0.059 |
| Char SVM | 0.915 | 0.056 |
| ViSoBERT | 0.920 | 0.053 |
| PhoBERT | 0.911 | 0.071 |

On `test_clean` itself, removing ViSoBERT changes nothing measurable. It is kept for robustness. In a stress test, the accented test messages had their accents removed, and chat abbreviations were substituted ("không" → "ko", "chuyển khoản" → "ck"). The ensemble with both transformers kept recall and the false-positive rate closest to the original: with PhoBERT alone the ensemble lost recall, with ViSoBERT alone it gained false positives.

| Stress test (ensemble fitted on the original messages) | Recall | False-positive rate |
|---|---|---|
| Accents removed (435 messages) — with both transformers | 0.960 → **0.954** | 0.060 → **0.070** |
| — PhoBERT only | 0.960 → 0.934 | 0.056 → 0.077 |
| — ViSoBERT only (previous ensemble) | 0.940 → 0.934 | 0.070 → 0.151 |
| Accents removed + abbreviations (792 messages) — with both transformers | 0.926 → 0.905 | 0.053 → 0.066 |
| — PhoBERT only | 0.931 → 0.887 | 0.055 → 0.066 |
| — ViSoBERT only (previous ensemble) | 0.926 → 0.926 | 0.073 → 0.103 |

**The models have different strengths.**
PR-AUC by type of message (higher is better):

| Messages | PhoBERT | ViSoBERT | Char SVM | Word LR |
|---|---|---|---|---|
| Without placeholder tokens (191) | **0.98** | 0.87 | 0.66 | 0.72 |
| Short, under 40 tokens (254) | **0.94** | 0.88 | 0.84 | 0.87 |
| Long, 100 tokens or more (111) | 0.79 | 0.54 | 0.83 | **0.86** |
| Unaccented (462) | **0.91** | 0.83 | 0.85 | 0.88 |
| Accented, with accents removed (435) | **0.97** | 0.92 | 0.82 | 0.80 |

- **PhoBERT** is the strongest single model when there is no obvious keyword or placeholder token to rely on.
- **Word LR** is the best on long, formal impersonation scams.
- **The TF-IDF models depend on accents**: removing them drops their PR-AUC from about 0.96 to 0.80, while the transformers barely change (both were trained with unaccented copies).
- **ViSoBERT** is the weakest alone, but its mistakes differ from PhoBERT's (score correlation 0.89), which is what keeps the ensemble's recall up in the stress test.

**Teencode and abbreviations are rare in the data**: about 3% of messages are written in chat style, and only 12 use digit-for-letter obfuscation (`b0c0`, `t3le`), 1 of them in the test set. Robustness to these styles is therefore measured only through the stress test above, not on real examples.

## Limitations

- The ensemble weights and threshold are fitted on `test_clean`. The cross-fitted numbers above are an honest estimate, but picking the best of about 10 methods (and the PhoBERT run) on the same set is still slightly optimistic. A fully independent estimate needs newly collected messages, evaluated once.
- Prediction is limited to what is written in the message. Sender identity, link reputation and message history are not used.
- Name detection (`[NAME]`) is rule-based. It misses names at the start of a sentence after a capitalised kinship word ("Chị Lan ơi"), and it can occasionally tag Title-Cased words as names.
- Scam tactics change over time, so the models need retraining on newly collected data.

---

## Project structure

```text
scam_detector/
├── preprocessing.py   link and personal-data tagging, normalisation for TF-IDF, ViSoBERT and PhoBERT (word segmentation)
├── detector.py        loads the four models, scores a message, applies the stacking ensemble
└── config.py          stacking coefficients, scaler statistics, threshold, model locations
models/                trained TF-IDF vectorizers and classifiers (.joblib)
api.py                 FastAPI service: POST /predict, GET /health
main.py                command-line interface
docs/                  web interface (GitHub Pages)
```

The fine-tuned transformers are hosted on the Hugging Face Hub and downloaded on first use, at the commit pinned in `scam_detector/config.py`:

- ViSoBERT: [`dohaduyphong/visobert-scam-sms-vn`](https://huggingface.co/dohaduyphong/visobert-scam-sms-vn) (local copy: `models/visobert/`)
- PhoBERT: [`dohaduyphong/phobert-scam-sms-vn`](https://huggingface.co/dohaduyphong/phobert-scam-sms-vn) (local copy: `models/phobert_base_v2/`)

A local copy is used instead when it exists.

## Usage

```bash
pip install -r requirements.txt
python3 main.py "Tai khoan cua ban bi khoa, truy cap abc-vn.com de mo lai"   # CLI
uvicorn api:app --port 8000                                                  # API
```

`POST /predict` with `{"message": "..."}` returns the score of each model and the ensemble decision:

```json
{
  "proba_word_tfidf_logreg": 0.842,
  "proba_char_tfidf_svm": 0.972,
  "proba_visobert": 0.924,
  "proba_phobert": 0.956,
  "confidence_scam": 0.995,
  "threshold": 0.635,
  "label": "scam"
}
```

Production setup is described in [DEPLOY.md](DEPLOY.md).

## Acknowledgements

The dataset is *Vietnamese SMS Dataset with Quality Assurance* by Tran N. T. T., Le H. K., Nguyen M. T., Nguyen V. T. and Mai H. D. (IEEE Access, 2026), released under CC BY 4.0.
The base transformers are [ViSoBERT](https://huggingface.co/uitnlp/visobert) by UIT NLP and [PhoBERT](https://huggingface.co/vinai/phobert-base-v2) by VinAI Research; word segmentation uses [pyvi](https://github.com/trungtv/pyvi).
