#!/usr/bin/env python3
"""Download a GGUF model for the local LLM server.

Usage:
    python3 download_model.py [small|medium|large]

Downloads into this directory. Skips re-download if the file already
exists (matches the harness's own skip-if-cached philosophy).
"""
import os
import sys

from huggingface_hub import hf_hub_download

from models import DEFAULT_TIER, MODELS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TIER
    if tier not in MODELS:
        print(f"Unknown tier '{tier}'. Choose from: {list(MODELS)}")
        sys.exit(1)
    spec = MODELS[tier]
    target = os.path.join(HERE, spec["filename"])
    if os.path.exists(target):
        print(f"already downloaded: {target}")
        return target
    print(f"downloading {tier} tier: {spec['repo_id']}/{spec['filename']}")
    print(f"  ({spec['note']})")
    path = hf_hub_download(repo_id=spec["repo_id"],
                            filename=spec["filename"],
                            local_dir=HERE)
    print("downloaded to:", path)
    return path


if __name__ == "__main__":
    main()
