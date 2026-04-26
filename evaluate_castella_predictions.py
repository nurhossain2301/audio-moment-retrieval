#!/usr/bin/env python3
"""Compute full evaluation metrics for CASTELLA audio moment retrieval predictions.

Computes Recall@IoU (R@0.3, R@0.5, R@0.7) at different top-k values.
Saves a summary JSON with all metrics.

Usage:
    python evaluate_castella_predictions.py \
        --predictions results/castella_baseline_predictions.json \
        --output results/castella_evaluation_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from lighthouse.common.utils.span_utils import temporal_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CASTELLA predictions")
    parser.add_argument(
        "--predictions",
        type=str,
        default="results/castella_baseline_predictions.json",
        help="Path to predictions JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/castella_evaluation_metrics.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--top-ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Top-k values to compute recall for.",
    )
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="IoU thresholds for recall computation.",
    )
    return parser.parse_args()


def load_predictions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected predictions list at {path}")
    return data


def compute_recall_at_iou(
    predictions: List[Dict[str, Any]],
    top_ks: List[int],
    iou_thresholds: List[float],
) -> Dict[str, Dict[str, float]]:
    """Compute Recall@IoU for multiple top-k values.
    
    Returns dict of format:
        {
            "R@0.3": {"top_1": ..., "top_5": ..., ...},
            "R@0.5": {...},
            ...
        }
    """
    results = {f"R@{int(th*100)}": {} for th in iou_thresholds}

    for top_k in top_ks:
        hit_counts = {f"R@{int(th*100)}": 0 for th in iou_thresholds}
        total = len(predictions)

        for item in predictions:
            pred_windows = item.get("pred_relevant_windows", [])[:top_k]
            gt_timestamps = item.get("gt_timestamps", [])

            if not pred_windows or not gt_timestamps:
                continue

            # Convert predictions to spans
            pred_spans = torch.tensor(
                [[float(p[0]), float(p[1])] for p in pred_windows],
                dtype=torch.float32,
            )
            # Convert ground truth to spans
            gt_spans = torch.tensor(
                [[float(ts[0]), float(ts[1])] for ts in gt_timestamps],
                dtype=torch.float32,
            )

            if pred_spans.numel() == 0 or gt_spans.numel() == 0:
                continue

            # Compute IoU
            iou_matrix, _ = temporal_iou(pred_spans, gt_spans)
            best_iou = float(iou_matrix.max().item())

            # Check against thresholds
            for th in iou_thresholds:
                if best_iou >= th:
                    hit_counts[f"R@{int(th*100)}"] += 1

        # Store results for this top-k
        for th in iou_thresholds:
            metric_key = f"R@{int(th*100)}"
            recall_value = (
                hit_counts[metric_key] / total if total > 0 else 0.0
            )
            results[metric_key][f"top_{top_k}"] = recall_value

    return results


def compute_additional_stats(
    predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute additional statistics about predictions."""
    total_queries = len(predictions)
    
    # Count predictions with at least one result
    with_results = sum(
        1 for p in predictions 
        if p.get("pred_relevant_windows") and len(p["pred_relevant_windows"]) > 0
    )
    
    # Average number of predictions per query
    avg_preds = np.mean([
        len(p.get("pred_relevant_windows", []))
        for p in predictions
    ])
    
    # Average confidence (first prediction score)
    scores = [
        float(p["pred_relevant_windows"][0][2])
        for p in predictions
        if p.get("pred_relevant_windows") and len(p["pred_relevant_windows"]) > 0
    ]
    avg_confidence = np.mean(scores) if scores else 0.0
    median_confidence = np.median(scores) if scores else 0.0
    
    return {
        "total_queries": total_queries,
        "queries_with_results": with_results,
        "coverage": with_results / total_queries if total_queries > 0 else 0.0,
        "avg_predictions_per_query": float(avg_preds),
        "avg_confidence": float(avg_confidence),
        "median_confidence": float(median_confidence),
    }


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")

    print("\nComputing recall metrics...")
    recall_results = compute_recall_at_iou(
        predictions,
        top_ks=args.top_ks,
        iou_thresholds=args.iou_thresholds,
    )

    print("Computing additional statistics...")
    stats = compute_additional_stats(predictions)

    # Combine results
    metrics = {
        "timestamp": str(Path(args.output).parent),
        "stats": stats,
        "recall_metrics": recall_results,
    }

    print(f"\nSaving metrics to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("CASTELLA BASELINE EVALUATION SUMMARY")
    print("="*60)
    print(f"Total queries: {stats['total_queries']}")
    print(f"Queries with results: {stats['queries_with_results']}")
    print(f"Coverage: {stats['coverage']:.4f}")
    print(f"Avg predictions per query: {stats['avg_predictions_per_query']:.2f}")
    print(f"Avg confidence: {stats['avg_confidence']:.4f}")
    print(f"Median confidence: {stats['median_confidence']:.4f}")
    print("\nRecall@IoU Metrics:")
    for threshold in sorted(recall_results.keys()):
        print(f"\n  {threshold}:")
        for top_k in sorted(recall_results[threshold].keys(), key=lambda x: int(x.split('_')[1])):
            value = recall_results[threshold][top_k]
            print(f"    {top_k}: {value:.4f}")


if __name__ == "__main__":
    main()
