"""
One-time script: push the fine-tuned PhoBERT model (weights, tokenizer,
label_mapping.json) to its own Hugging Face Model repo.

Run this from a machine that has the real
models/phobert_scam_sms/ folder (from phobert_scam_sms.ipynb's save
step) -- NOT from the Space itself.

Usage:
    pip install huggingface_hub
    huggingface-cli login              # one-time auth, needs a write token
    python3 push_phobert_to_hub.py yourname/scam-sms-phobert

Why a separate Model repo instead of committing into the Space: a
Space's own git repo is capped at 1GB, too tight for PhoBERT (~500MB+)
alongside everything else. Model repos have no such cap. The Space
downloads this repo at container startup via PHOBERT_SOURCE
(see scam_detector/config.py) instead of shipping it in its own image.
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

    print("\nDone. Set this on your Space (Settings -> Variables and secrets):")
    print(f"  PHOBERT_SOURCE = {repo_id}")


if __name__ == "__main__":
    main()
