#!/usr/bin/env python3
"""Generate baseline CASTELLA audio moment retrieval predictions.

This script uses the official CLAP + QD-DETR CASTELLA baseline checkpoint
and the precomputed CLAP audio features stored under `features/castella/`.

Example:
    python generate_castella_baseline_results.py \
        --dataset-json CASTELLA/json/en/test.json \
        --audio-feat-dir features/castella/clap \
        --output results/castella_baseline_predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import easydict
import numpy as np
import torch

from lighthouse.common.utils.basic_utils import l2_normalize_np_array
from lighthouse.common.utils.span_utils import temporal_iou
from lighthouse.models import QDDETRPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CASTELLA baseline audio moment retrieval predictions"
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default="CASTELLA/json/en/test.json",
        help="Path to CASTELLA JSON dataset split.",
    )
    parser.add_argument(
        "--audio-feat-dir",
        type=str,
        default="features/castella/clap",
        help="Directory with precomputed CASTELLA CLAP audio .npz files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="lighthouse/gradio_demo/weights/clap_qd_detr_finetuned.ckpt",
        help="Baseline model checkpoint path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/castella_baseline_predictions.json",
        help="Output predictions JSON path.",
    )
    parser.add_argument(
        "--jsonl-output",
        type=str,
        default="results/castella_baseline_predictions.jsonl",
        help="Official-style JSONL output path for Lighthouse/DCASE evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked window predictions to keep per query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="If > 0, limit the number of queries for a quick test run.",
    )
    return parser.parse_args()


def load_castella_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected CASTELLA JSON list at {path}")
    return data


def write_submission_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            submission_row = {
                "qid": row["qid"],
                "query": row["query"],
                "duration": row["duration"],
                "vid": row["vid"],
                "pred_relevant_windows": row["pred_relevant_windows"],
            }
            f.write(json.dumps(submission_row) + "\n")


def load_audio_features(audio_feat_dir: str, yid: str, max_frames: Optional[int] = None) -> np.ndarray:
    path = Path(audio_feat_dir) / f"{yid}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing audio feature file: {path}")

    data = np.load(path)
    if "features" not in data:
        raise ValueError(f"Audio feature file {path} does not contain key 'features'")

    features = data["features"].astype(np.float32)
    if max_frames is not None:
        features = features[:max_frames]

    return l2_normalize_np_array(features)


def make_audio_inputs(audio_feats: np.ndarray, device: str) -> Dict[str, torch.Tensor]:
    audio_feats_t = torch.from_numpy(audio_feats).to(device)
    n_frames = audio_feats_t.shape[0]

    audio_mask = torch.ones((1, n_frames), dtype=torch.long, device=device)
    tef_st = torch.arange(0, n_frames, dtype=torch.float32, device=device) / float(n_frames)
    tef_ed = tef_st + 1.0 / float(n_frames)
    video_feats = torch.stack([tef_st, tef_ed], dim=1).unsqueeze(0)

    return {
        "video_feats": video_feats,
        "video_mask": audio_mask,
        "audio_feats": audio_feats_t.unsqueeze(0),
    }


def compute_recall_at_iou(
    predictions: List[Dict[str, Any]],
    thresholds: List[float],
    top_k: int,
) -> Dict[str, float]:
    hit_counts = {f"R1@{int(th*100)}": 0 for th in thresholds}
    total = len(predictions)

    for item in predictions:
        pred_windows = item["pred_relevant_windows"][:top_k]
        pred_spans = torch.tensor([p[:2] for p in pred_windows], dtype=torch.float32)
        gt_spans = torch.tensor(item["gt_timestamps"], dtype=torch.float32)

        if pred_spans.numel() == 0 or gt_spans.numel() == 0:
            continue

        iou_matrix, _ = temporal_iou(pred_spans, gt_spans)
        best_iou = float(iou_matrix.max().item())
        for th in thresholds:
            if best_iou >= th:
                hit_counts[f"R1@{int(th*100)}"] += 1

    return {metric: hit_counts[metric] / total if total else 0.0 for metric in hit_counts}


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading model checkpoint:", args.checkpoint)
    with torch.serialization.safe_globals([easydict.EasyDict]):
        model = QDDETRPredictor(args.checkpoint, device=args.device, feature_name="clap")
    model._model.eval()

    dataset = load_castella_json(args.dataset_json)
    print(f"Loaded {len(dataset)} CASTELLA videos from {args.dataset_json}")

    predictions: List[Dict[str, Any]] = []
    query_count = 0

    for entry in dataset:
        yid = entry.get("yid")
        if yid is None:
            raise KeyError("CASTELLA entry is missing 'yid'")

        try:
            audio_feats = load_audio_features(args.audio_feat_dir, yid)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
            continue

        inputs = make_audio_inputs(audio_feats, args.device)
        moments = entry.get("moments", [])
        for moment_idx, moment in enumerate(moments, start=1):
            if args.limit > 0 and query_count >= args.limit:
                break

            query_text = moment.get("local_caption") or moment.get("caption") or ""
            if not query_text:
                print(f"WARNING: empty query for video {yid} moment {moment_idx}")
                continue

            qid = f"{yid}_{moment_idx}"
            print(f"Predicting {qid}: {query_text}")

            with torch.no_grad():
                prediction = model.predict(query_text, inputs)

            if prediction is None:
                print(f"WARNING: prediction returned None for {qid}")
                continue

            prediction["pred_relevant_windows"] = [
                [float(st), float(ed), float(score)]
                for st, ed, score in prediction["pred_relevant_windows"][: args.top_k]
            ]

            result = {
                "qid": qid,
                "vid": yid,
                "yid": yid,
                "query": query_text,
                "duration": entry.get("duration"),
                "gt_timestamps": moment.get("timestamps", []),
                "relevant_windows": moment.get("timestamps", []),
                "pred_relevant_windows": prediction["pred_relevant_windows"],
                "pred_saliency_scores": [float(x) for x in prediction.get("pred_saliency_scores", [])],
            }
            predictions.append(result)
            query_count += 1

        if args.limit > 0 and query_count >= args.limit:
            break

    print(f"Saving predictions to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saving official-style predictions to {args.jsonl_output}")
    Path(args.jsonl_output).parent.mkdir(parents=True, exist_ok=True)
    write_submission_jsonl(args.jsonl_output, predictions)

    if predictions:
        metrics = compute_recall_at_iou(predictions, [0.3, 0.5, 0.7], top_k=1)
        print("Evaluation summary")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"  total_queries: {len(predictions)}")


if __name__ == "__main__":
    main()
