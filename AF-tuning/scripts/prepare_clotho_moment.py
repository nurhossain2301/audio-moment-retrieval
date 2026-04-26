#!/usr/bin/env python3
"""Prepare Clotho-Moment as instruction data for AF-Next.

The Hugging Face Clotho-Moment snapshot stores raw audio in tar shards. This
script reads Lighthouse's JSONL annotations, finds the matching tar member, and
writes instruction-style JSONL records. With --extract-audio, it also extracts
and resamples the needed audio to mono 16 kHz WAV.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROMPT_TEMPLATE = (
    "You are solving audio moment retrieval. Given the audio and the query, "
    "return only JSON with key \"relevant_windows\". Each window must be "
    "[start_seconds, end_seconds]. Query: {query}"
)


@dataclass(frozen=True)
class AudioLocation:
    tar_path: Path
    member_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Clotho-Moment for AF-Next")
    parser.add_argument(
        "--raw-root",
        default="raw_datasets/clotho_moment",
        help="Downloaded HF Clotho-Moment root with train/valid/test tar shards.",
    )
    parser.add_argument(
        "--annotation-root",
        default="lighthouse/data/clotho_moment",
        help="Directory containing clotho_moment_*_release.jsonl files.",
    )
    parser.add_argument(
        "--output-root",
        default="AF-tuning/data/clotho_moment",
        help="Where prepared records and optional extracted audio are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Splits to prepare.",
    )
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Extract/resample referenced audio to mono 16 kHz WAV.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum records per split for smoke tests.",
    )
    return parser.parse_args()


def annotation_path(annotation_root: Path, split: str) -> Path:
    return annotation_root / f"clotho_moment_{split}_release.jsonl"


def raw_split_dir(raw_root: Path, split: str) -> Path:
    return raw_root / ("valid" if split == "val" else split)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def member_name_from_vid(vid: str) -> str:
    # Release vids look like "Venice_4.0_64.0"; tar members use "Venice_40_640.wav".
    prefix, start, end = vid.rsplit("_", 2)
    start_i = int(round(float(start) * 10))
    end_i = int(round(float(end) * 10))
    return f"{prefix}_{start_i}_{end_i}.wav"


def build_audio_index(raw_dir: Path, required_members: set[str]) -> dict[str, AudioLocation]:
    index: dict[str, AudioLocation] = {}
    remaining = set(required_members)
    for tar_path in sorted(raw_dir.glob("*.tar")):
        if not remaining:
            break
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name in remaining:
                    index[member.name] = AudioLocation(tar_path, member.name)
                    remaining.remove(member.name)
                    if not remaining:
                        break
    return index


def extract_member_to_wav(location: AudioLocation, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(location.tar_path, "r") as tar:
        source = tar.extractfile(location.member_name)
        if source is None:
            raise FileNotFoundError(f"Missing {location.member_name} in {location.tar_path}")
        wav_bytes = source.read()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(cmd, input=wav_bytes, check=True)


def build_answer(relevant_windows: list[list[float]]) -> str:
    rounded = [[round(float(st), 3), round(float(ed), 3)] for st, ed in relevant_windows]
    return json.dumps({"relevant_windows": rounded}, separators=(",", ":"))


def prepare_split(args: argparse.Namespace, split: str) -> None:
    raw_root = Path(args.raw_root)
    annotation_root = Path(args.annotation_root)
    output_root = Path(args.output_root)
    records_path = output_root / f"{split}.jsonl"
    audio_root = output_root / "audio" / split

    raw_dir = raw_split_dir(raw_root, split)
    rows = []
    required_members = set()
    for row in iter_jsonl(annotation_path(annotation_root, split)):
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break
        member_name = member_name_from_vid(row["vid"])
        rows.append((row, member_name))
        required_members.add(member_name)

    audio_index = build_audio_index(raw_dir, required_members)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    missing = 0
    written = 0
    with records_path.open("w", encoding="utf-8") as out_f:
        for row, member_name in rows:
            location = audio_index.get(member_name)
            if location is None:
                missing += 1
                continue

            audio_path = audio_root / member_name
            if args.extract_audio:
                extract_member_to_wav(location, audio_path)

            record = {
                "qid": row["qid"],
                "split": split,
                "query": row["query"],
                "duration": row["duration"],
                "vid": row["vid"],
                "audio": str(audio_path if args.extract_audio else location.tar_path),
                "tar_member": location.member_name,
                "prompt": PROMPT_TEMPLATE.format(query=row["query"]),
                "answer": build_answer(row["relevant_windows"]),
                "relevant_windows": row["relevant_windows"],
            }
            out_f.write(json.dumps(record) + "\n")
            written += 1

    print(f"{split}: wrote {written} records to {records_path}; missing audio: {missing}")


def main() -> None:
    args = parse_args()
    for split in args.splits:
        prepare_split(args, split)


if __name__ == "__main__":
    main()
