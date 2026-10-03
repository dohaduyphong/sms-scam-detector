# Deployment (no Docker)

```text
GitHub (dohaduyphong/sms-scam-detector)
  │ git clone / git pull
  ▼
Ubuntu VPS managed by xCloud
  │
  ├── Python .venv  ──  Uvicorn (api:app) on 127.0.0.1:8000  ──  systemd --user: sms-scam-detector
  │
  └── Nginx (xCloud, SSL)  https://api.duyphong.info  ──proxy──▶  127.0.0.1:8000

Frontend: GitHub Pages (https://dohaduyphong.github.io/sms-scam-detector/, from docs/)
          calls https://api.duyphong.info/predict
```

| Item | Value |
|---|---|
| ASGI app | `api:app` (`api.py`, repo root; working directory must be the repo root) |
| Internal address | `127.0.0.1:8000` (never exposed publicly) |
| Public URL | `https://api.duyphong.info` |
| Endpoints | `GET /health` → `{"status":"ok"}`, `POST /predict` `{"message": "..."}` |
| Python | 3.12 (Ubuntu 24.04 default) or 3.11. Tested with both. |
| RAM | ~1–1.5 GB for the process (PhoBERT fp32 + torch). Use 1 Uvicorn worker. |
| Process manager | systemd **user** service `sms-scam-detector` of the deploy user (no sudo; linger enabled so it starts at boot) |

## 1. Models

All four TF-IDF/LogReg/SVM models are tracked in git, so `git clone` gets them.
**PhoBERT is not in git** (`models/phobert_full/` is gitignored, ~517 MB).

| Model | Path | In git? |
|---|---|---|
| Word TF-IDF vectorizer | `models/word_tfidf_vectorizer_full.joblib` | yes |
| Logistic Regression | `models/logreg_model_full.joblib` | yes |
| Char TF-IDF vectorizer | `models/char_tfidf_vectorizer_full.joblib` | yes |
| SVM (calibrated) | `models/svm_model_full.joblib` | yes |
| PhoBERT weights, tokenizer, `label_mapping.json` | `models/phobert_full/` | **no** |

**Production source of PhoBERT: the Hugging Face model repo
[`dohaduyphong/phobert-scam-sms-vn`](https://huggingface.co/dohaduyphong/phobert-scam-sms-vn).**
The systemd template sets `PHOBERT_SOURCE=dohaduyphong/phobert-scam-sms-vn`; on first start
the server downloads it (~540 MB) into the Hugging Face cache (`HF_HOME`, default
`~/.cache/huggingface`) and reuses the cache afterwards.

- The repo is **public: no token, no `huggingface-cli login`, no secret needed.**
- Its files are byte-identical to the local `models/phobert_full/` (verified; same predictions).
- `label_mapping.json` is fetched from the same repo (`scam_detector/detector.py`).
- Never commit the 517 MB model to git. To publish a retrained PhoBERT, use
  `push_phobert_to_hub.py` (needs a write token on the machine that pushes, not on the server).

Without `PHOBERT_SOURCE`, `scam_detector/config.py` falls back to the local folder
`models/phobert_full/` (local development; or copy it to the server with `rsync` as an
offline alternative).

`models/phobert/` (older PhoBERT) and the `*.joblib` files without `_full` are not used by the API.

## 2. Install

All steps run as the deploy user, no sudo. Requires `python3` (3.11/3.12) with `venv`,
`git` and `curl` on the server; if missing, an admin installs them once
(`apt install python3 python3-venv git curl`).

```bash
python3 --version && python3 -m venv --help >/dev/null && git --version && curl --version | head -1
git clone https://github.com/dohaduyphong/sms-scam-detector.git <APP_DIR>
cd <APP_DIR>
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# --extra-index-url: CPU-only torch (PyPI's Linux torch bundles CUDA, several GB)
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# PhoBERT is downloaded from Hugging Face on first start (section 1)
```

Keep `scikit-learn==1.6.1`: the `.joblib` models were trained with it.

## 3. Environment variables

Set in the systemd unit. No secrets or tokens are needed.

| Variable | Default | Purpose |
|---|---|---|
| `PHOBERT_SOURCE` | unset → `<APP_DIR>/models/phobert_full` | **production: `dohaduyphong/phobert-scam-sms-vn`** (set in template). Local folder or HF repo id |
| `HF_HOME` | `~/.cache/huggingface` | optional: where the PhoBERT download is cached |

CORS origins are in `scam_detector/config.py` (`CORS_ORIGINS`).

## 4. Run

Manual test:
```bash
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

systemd **user** service (default). Runs as the deploy user, no sudo. Template:
`deploy/sms-scam-detector.service.example` (replace `<APP_DIR>`).

```bash
mkdir -p ~/.config/systemd/user
cp deploy/sms-scam-detector.service.example ~/.config/systemd/user/sms-scam-detector.service
nano ~/.config/systemd/user/sms-scam-detector.service
systemctl --user daemon-reload
systemctl --user enable --now sms-scam-detector
systemctl --user status sms-scam-detector
journalctl --user -u sms-scam-detector -f   # model loading takes ~10-60s
```

- Boot: the unit is `WantedBy=default.target` and lingering is enabled for the deploy user
  (`loginctl show-user $USER -p Linger` → `Linger=yes`), so it starts without a login.
  Enabling linger is a one-time admin step (`loginctl enable-linger <user>`); already done
  on the production server.
- `Failed to connect to bus` from `systemctl --user` / `journalctl --user` means the shell has
  no user session environment (common after `su`/`sudo -u`, cron, some SSH setups). Fix:
  ```bash
  export XDG_RUNTIME_DIR=/run/user/$(id -u)
  ```
  `deploy/update.sh` sets this automatically when it is missing.

Optional, not the default: a system-level unit (needs root) — copy the template to
`/etc/systemd/system/`, add `User=`/`Group=` under `[Service]`, set
`WantedBy=multi-user.target`, manage it with `sudo systemctl ...`, and run updates with
`SYSTEMD_SCOPE=system deploy/update.sh`.

## 5. Nginx / xCloud

In xCloud, point `api.duyphong.info` to a reverse proxy on `http://127.0.0.1:8000`
and let xCloud handle SSL. Reference config (do not overwrite xCloud's files):
`deploy/nginx-api.duyphong.info.conf.example`.

## 6. Test

```bash
curl http://127.0.0.1:8000/health
curl https://api.duyphong.info/health
curl -X POST https://api.duyphong.info/predict \
  -H 'Content-Type: application/json' \
  -d '{"message":"Chuc mung ban da trung thuong 50 trieu, lien he 0912345678 de nhan qua"}'
# CORS preflight from the GitHub Pages origin must return access-control-allow-origin
curl -si -X OPTIONS https://api.duyphong.info/predict \
  -H 'Origin: https://dohaduyphong.github.io' -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: content-type' | grep -i access-control-allow-origin
```

## 7. Update

```bash
cd <APP_DIR> && deploy/update.sh
```

`git pull --ff-only` → `pip install` only if `requirements.txt` changed →
`systemctl --user restart sms-scam-detector` → wait for `/health` (up to 180s). No sudo.
It never runs `git reset`/`git clean`; it stops before pulling if tracked files have local
changes or the user unit is not found.

Options: `FORCE_PIP=1`, `FORCE_RESTART=1`, `HEALTH_TIMEOUT=<s>`, `SERVICE_NAME=<unit>`.
With the optional system-level unit, use `SYSTEMD_SCOPE=system`; it restarts via
`sudo -n systemctl restart` (never prompts, so it needs a matching sudoers rule).

If PhoBERT itself changes, push it to the HF repo and restart; to force a fresh download,
remove `~/.cache/huggingface/hub/models--dohaduyphong--phobert-scam-sms-vn` first.
`git pull` never updates the model.
