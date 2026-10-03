"""
One-time script: push the fine-tuned PhoBERT model (weights, tokenizer,
label_mapping.json) to its own Hugging Face Model repo.

Run this from a machine that has the trained models/phobert_full/
folder -- not from the production server.

Usage:
    pip install huggingface_hub
    huggingface-cli login              # one-time auth, needs a write token
    python3 push_phobert_to_hub.py yourname/scam-sms-phobert

Why a separate Model repo: PhoBERT (~500MB+) is too large for the
GitHub repo (models/phobert_full/ is gitignored). The production server
downloads it from this repo at startup via PHOBERT_SOURCE (see
scam_detector/config.py, deploy/sms-scam-detector.service.example and
DEPLOY.md). Production uses dohaduyphong/phobert-scam-sms-vn.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

PHOBERT_DIR = Path(__file__).resolve().parent / "models" / "phobert_full"


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 push_phobert_to_hub.py <yourname>/<repo-name>")
        sys.exit(1)
    repo_id = sys.argv[1]

    if not PHOBERT_DIR.exists():
        print(f"Not found: {PHOBERT_DIR}")
        print("Copy your trained models/phobert_full/ folder here first.")
        sys.exit(1)

    required = {"config.json", "label_mapping.json"}
    present = {p.name for p in PHOBERT_DIR.iterdir()}
    missing = required - present
    if missing:
        print(f"models/phobert_full/ is missing: {', '.join(missing)}")
        sys.exit(1)

    print(f"Creating (or reusing) model repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading {PHOBERT_DIR} -> {repo_id} ...")
    api = HfApi()
    api.upload_folder(
        folder_path=str(PHOBERT_DIR),
        repo_id=repo_id,
        repo_type="model",
    )

    print("\nDone. On the server, set in the systemd unit and restart:")
    print(f"  Environment=PHOBERT_SOURCE={repo_id}")


if __name__ == "__main__":
    main()
