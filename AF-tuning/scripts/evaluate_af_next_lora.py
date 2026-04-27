#!/usr/bin/env python3
"""Evaluate an AF-Next LoRA adapter on Clotho-Moment JSONL records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AF-Next LoRA on Clotho-Moment")
    parser.add_argument("--model-path", default="AF-tuning/models/audio-flamingo-next-hf")
    parser.add_argument("--adapter-path", default="AF-tuning/outputs/af-next-clotho-lora")
    parser.add_argument("--test-jsonl", default="AF-tuning/data/clotho_moment/test.jsonl")
    parser.add_argument("--output-jsonl", default="AF-tuning/outputs/af-next-clotho-lora/test_predictions.jsonl")
    parser.add_argument("--metrics-json", default="AF-tuning/outputs/af-next-clotho-lora/test_metrics.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


def load_rows(path: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = Path(row["audio"])
            if not audio.exists():
                raise FileNotFoundError(f"Missing audio for qid={row.get('qid')}: {audio}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def conversation_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": row["prompt"]},
                {"type": "audio", "path": row["audio"]},
            ],
        }
    ]


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_windows(text: str, duration: float) -> tuple[list[list[float]], bool]:
    payload = extract_json_object(text)
    windows: Any = None if payload is None else payload.get("relevant_windows")

    if not isinstance(windows, list):
        matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
        windows = [[float(start), float(end)] for start, end in matches]

    parsed: list[list[float]] = []
    if isinstance(windows, list):
        for window in windows:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                continue
            try:
                start = float(window[0])
                end = float(window[1])
            except (TypeError, ValueError):
                continue
            start = max(0.0, min(duration, start))
            end = max(0.0, min(duration, end))
            if end > start:
                parsed.append([round(start, 3), round(end, 3)])

    parsed.sort(key=lambda item: (item[0], item[1]))
    return parsed, bool(parsed)


def window_iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def best_iou(predicted: list[list[float]], gold: list[list[float]]) -> float:
    if not predicted or not gold:
        return 0.0
    return max(window_iou(pred, target) for pred in predicted for target in gold)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [record["best_iou"] for record in records]
    parsed = [record["parse_ok"] for record in records]
    return {
        "num_examples": len(records),
        "parse_rate": sum(parsed) / len(parsed) if parsed else 0.0,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "recall_at_iou_0.3": sum(iou >= 0.3 for iou in ious) / len(ious) if ious else 0.0,
        "recall_at_iou_0.5": sum(iou >= 0.5 for iou in ious) / len(ious) if ious else 0.0,
        "recall_at_iou_0.7": sum(iou >= 0.7 for iou in ious) / len(ious) if ious else 0.0,
        "avg_predicted_windows": (
            sum(len(record["predicted_windows"]) for record in records) / len(records) if records else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AF-Next evaluation.")

    rows = load_rows(args.test_jsonl, args.limit_samples)
    output_path = Path(args.output_jsonl)
    metrics_path = Path(args.metrics_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to("cuda")
    model.eval()

    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as out_f, torch.inference_mode():
        for row in tqdm(rows, desc="Evaluating"):
            batch = processor.apply_chat_template(
                [conversation_for(row)],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                processor_kwargs={"padding": True},
            ).to(model.device)
            if "input_features" in batch:
                batch["input_features"] = batch["input_features"].to(model.dtype)

            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
            )
            prompt_len = batch["input_ids"].shape[1]
            completion = generated[:, prompt_len:]
            response = processor.batch_decode(
                completion,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            predicted_windows, parse_ok = parse_windows(response, float(row["duration"]))
            gold_windows = [[float(start), float(end)] for start, end in row["relevant_windows"]]
            record = {
                "qid": row["qid"],
                "vid": row.get("vid"),
                "query": row["query"],
                "duration": row["duration"],
                "gold_windows": gold_windows,
                "predicted_windows": predicted_windows,
                "best_iou": best_iou(predicted_windows, gold_windows),
                "parse_ok": parse_ok,
                "response": response,
            }
            records.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    metrics = summarize(records)
    metrics.update(
        {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "test_jsonl": args.test_jsonl,
            "output_jsonl": args.output_jsonl,
        }
    )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
