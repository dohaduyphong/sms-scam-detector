# Vietnamese SMS Scam Detector

A system that classifies Vietnamese SMS messages as **scam** or **ham (legitimate)**.

Three models read the same message in different ways, and a stacking ensemble combines what they see:

| Model | What it looks at |
|---|---|
| **Word TF-IDF + Logistic Regression** | Words and two-word phrases ("chuyển khoản", "khóa tài khoản") |
| **Character TF-IDF + Linear SVM** | 3–5 character fragments. These still match when a word is misspelled, unaccented or deliberately obfuscated (`Gja'`, `1OO%`, `T-D-T-C`) |
| **ViSoBERT** (fine-tuned transformer) | The meaning of the whole sentence. It works on short messages and messages without obvious keywords |

The ensemble catches **93% of scam messages** and flags **7% of legitimate messages** by mistake. These numbers come from a held-out test set that includes deliberately hard cases ([Results](#results)).

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
   └─► Unicode NFC, keep case/punct ─► ViSoBERT tokenizer      ─► ViSoBERT            ─► s_bert
                                                    │
                     2. standardise each score:  z_i = (s_i − mean_i) / std_i
                     3. stacking:               score = b + w₁·z_word + w₂·z_char + w₃·z_bert
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

The two TF-IDF models then lowercase the text and strip punctuation. The placeholder tokens are protected while this happens, so `[MONEY]` stays a single token. ViSoBERT gets the text with case and punctuation kept, only Unicode-normalised (NFC).

### 2–3. Combining the three models (`scam_detector/detector.py`)

The three models produce scores on different scales, so each one is first converted to log-odds:

- Logistic Regression: `decision_function` (log-odds)
- Linear SVM: `logit(P(scam))` from the calibrated SVM
- ViSoBERT: `logit_scam − logit_ham`

Each score is then standardised to a z-score. A logistic-regression meta-model (**stacking**) combines the three z-scores. Its coefficients come from the ensemble study described below:

| | Word LR | Char SVM | ViSoBERT |
|---|---|---|---|
| Stacking coefficient | 2.33 | 0.86 | 1.05 |

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
   └─► ViSoBERT fine-tune        │
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

### Model 3 — ViSoBERT

`uitnlp/visobert` is an XLM-RoBERTa model pre-trained on Vietnamese social-media text. It is fine-tuned here for binary classification. The changes below were made after studying where the earlier version failed:

| Choice | Why |
|---|---|
| **Placeholder tokens added as special tokens.** Their embeddings start from a related word: `[MONEY]` from "tiền", `[PHONE]` from "điện thoại" | The original tokenizer split `[MONEY]` into five meaningless pieces (`▁[ · M · ONE · Y · ]`), which hid the strongest scam signal in the data |
| **Best checkpoint chosen by scam-class PR-AUC** on `logit_scam − logit_ham` | That is exactly the score the ensemble uses. The earlier weighted-F1 criterion was dominated by the ham class, which makes up 73% of the data |
| **Validation set = 1/6 of train**, split by near-duplicate groups (`StratifiedGroupKFold`) | About 140 scam messages, enough to pick a checkpoint reliably. Near-duplicates cannot end up on both sides of the split |
| **Unaccented copies** of the training messages that have accents (train part only, after the split) | More than half of real SMS are written without diacritics, and every model was weaker on them |
| lr 3e-5, batch 16, up to 8 epochs, early stopping after 2 epochs without improvement, warmup 10%, weight decay 0.01, max length 256, fp16 on GPU | Converges in a few epochs on about 3–4k examples |

After these changes, ViSoBERT's recall on the hard test cases (at its default threshold) rose from 0.60 to 0.90.

### Ensemble study

All three models have already seen `train.csv`. Their scores on it would be overly optimistic, so the way to combine them is learned on **`test_clean`** instead. Cross-fitting keeps that estimate honest:

1. `test_clean` is split into 5 folds, grouped by near-duplicates.
2. For each fold, the combination method and its threshold are fitted on the other 4 folds and then used to predict that fold. Every message is therefore predicted by a model that never saw it.
3. **Threshold policy (catch scams first).** Among thresholds that keep recall ≥ 0.93, choose the one with the highest precision.
4. **Selection rule.** Keep the methods that reach the recall target, then pick the one with the highest **F2**. F2 weights recall about four times as much as precision. Ties are broken by PR-AUC.

Ten options were compared:
- each model on its own;
- majority voting (1 of 3, 2 of 3);
- average of z-scores, average of percentile ranks, maximum z-score;
- a grid search over weights;
- **stacking with logistic regression**, which was selected.

A drop-one study then checks that every model actually contributes.

---

## Results

Cross-fitted results on `test_clean` (897 messages, 250 scam):

| Method | Recall | Precision | F2 | False-positive rate | PR-AUC |
|---|---|---|---|---|---|
| **Stacking ensemble** | **0.932** | **0.835** | **0.911** | **0.071** | **0.943** |
| Average of z-scores | 0.932 | 0.820 | 0.907 | 0.079 | 0.940 |
| Char TF-IDF + SVM only | 0.932 | 0.766 | 0.893 | 0.110 | 0.908 |
| ViSoBERT only | 0.928 | 0.758 | 0.888 | 0.114 | 0.900 |
| Word TF-IDF + LR only | 0.924 | 0.760 | 0.886 | 0.113 | 0.932 |

The false-positive rate is the share of legitimate messages wrongly flagged as scam. 95% bootstrap confidence intervals for the ensemble: F2 [0.883, 0.936], false-positive rate [0.052, 0.093].

**Every model contributes.** Removing any one of them from the ensemble makes it worse:

| Removed | F2 | False-positive rate |
|---|---|---|
| — (all three) | 0.911 | 0.071 |
| Word LR | 0.903 | 0.082 |
| Char SVM | 0.903 | 0.088 |
| ViSoBERT | 0.899 | 0.104 |

**The three models have different strengths.**
PR-AUC by type of message (higher is better):

| Messages | ViSoBERT | Char SVM | Word LR |
|---|---|---|---|
| Without placeholder tokens (194) | **0.84** | 0.65 | 0.72 |
| Short, under 40 tokens (229) | **0.88** | 0.84 | 0.84 |
| Long, 100 tokens or more (115) | 0.54 | 0.83 | **0.86** |

- **ViSoBERT** is the best when there is no obvious keyword or placeholder token to rely on.
- **Word LR** is the best on long, formal impersonation scams.
- **Char SVM** is the most robust to obfuscated spelling.

**Hard cases** (389 hand-added messages): the ensemble reaches recall 0.93 and flags 9% of legitimate messages by mistake.

## Limitations

- The ensemble weights and threshold are fitted on `test_clean`. The cross-fitted numbers above are an honest estimate, but picking the best of about 10 methods on the same set is still slightly optimistic. A fully independent estimate needs newly collected messages, evaluated once.
- Prediction is limited to what is written in the message. Sender identity, link reputation and message history are not used.
- Name detection (`[NAME]`) is rule-based. It misses names at the start of a sentence after a capitalised kinship word ("Chị Lan ơi"), and it can occasionally tag Title-Cased words as names.
- Scam tactics change over time, so the models need retraining on newly collected data.

---

## Project structure

```text
scam_detector/
├── preprocessing.py   link and personal-data tagging, normalisation for TF-IDF and ViSoBERT
├── detector.py        loads the three models, scores a message, applies the stacking ensemble
└── config.py          stacking coefficients, scaler statistics, threshold, model locations
models/                trained TF-IDF vectorizers and classifiers (.joblib)
api.py                 FastAPI service: POST /predict, GET /health
main.py                command-line interface
docs/                  web interface (GitHub Pages)
```

The fine-tuned ViSoBERT model is hosted on the Hugging Face Hub at [`dohaduyphong/visobert-scam-sms-vn`](https://huggingface.co/dohaduyphong/visobert-scam-sms-vn) and is downloaded on first use. A local copy at `models/visobert/` is used instead when it exists.

## Usage

```bash
pip install -r requirements.txt
python3 main.py "Tai khoan cua ban bi khoa, truy cap abc-vn.com de mo lai"   # CLI
uvicorn api:app --port 8000                                                  # API
```

`POST /predict` with `{"message": "..."}` returns the score of each model and the ensemble decision:

```json
{
  "proba_word_tfidf_logreg": 0.876,
  "proba_char_tfidf_svm": 0.976,
  "proba_visobert": 0.999,
  "confidence_scam": 0.997,
  "threshold": 0.492,
  "label": "scam"
}
```

Production setup is described in [DEPLOY.md](DEPLOY.md).

## Acknowledgements

The dataset is *Vietnamese SMS Dataset with Quality Assurance* by Tran N. T. T., Le H. K., Nguyen M. T., Nguyen V. T. and Mai H. D. (IEEE Access, 2026), released under CC BY 4.0.
The base transformer is [ViSoBERT](https://huggingface.co/uitnlp/visobert) by UIT NLP.
