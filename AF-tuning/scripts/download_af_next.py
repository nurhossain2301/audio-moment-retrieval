#!/usr/bin/env python3
"""Download Audio Flamingo Next weights into the local AF-tuning folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AF-Next from Hugging Face")
    parser.add_argument(
        "--repo-id",
        default="nvidia/audio-flamingo-next-hf",
        help="Hugging Face model repository to download.",
    )
    parser.add_argument(
        "--local-dir",
        default="AF-tuning/models/audio-flamingo-next-hf",
        help="Where to save the model snapshot.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(local_dir),
        resume_download=True,
    )
    print(f"Downloaded {args.repo_id} to {local_dir}")


if __name__ == "__main__":
    main()
