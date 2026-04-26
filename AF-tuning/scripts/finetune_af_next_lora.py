#!/usr/bin/env python3
"""Supervised LoRA fine-tuning scaffold for AF-Next on AMR records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor, Trainer, TrainingArguments


class JsonlDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if not Path(row["audio"]).exists():
                        raise FileNotFoundError(
                            f"Audio path missing for {row['qid']}: {row['audio']}. "
                            "Run prepare_clotho_moment.py with --extract-audio first."
                        )
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class AFNextCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def _conversation(self, row: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": row["prompt"]},
                    {"type": "audio", "path": row["audio"]},
                ],
            }
        ]
        if include_answer:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["answer"]}],
                }
            )
        return conversation

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompt_batch = self.processor.apply_chat_template(
            [self._conversation(row, include_answer=False) for row in rows],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
        )
        full_batch = self.processor.apply_chat_template(
            [self._conversation(row, include_answer=True) for row in rows],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            padding=True,
        )

        labels = full_batch["input_ids"].clone()
        for i in range(labels.shape[0]):
            prompt_len = int(prompt_batch["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100
        labels[full_batch["attention_mask"] == 0] = -100
        full_batch["labels"] = labels
        if "input_features" in full_batch:
            full_batch["input_features"] = full_batch["input_features"].to(torch.bfloat16)
        return full_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AF-Next with LoRA")
    parser.add_argument("--model-path", default="AF-tuning/models/audio-flamingo-next-hf")
    parser.add_argument("--train-jsonl", default="AF-tuning/data/clotho_moment/train.jsonl")
    parser.add_argument("--eval-jsonl", default="AF-tuning/data/clotho_moment/val.jsonl")
    parser.add_argument("--output-dir", default="AF-tuning/outputs/af-next-clotho-lora")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--use-lora", action="store_true", help="Enable PEFT LoRA adapters.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--limit-train-samples",
        type=int,
        default=None,
        help="Optional train subset size for smoke tests.",
    )
    parser.add_argument(
        "--limit-eval-samples",
        type=int,
        default=None,
        help="Optional eval subset size for faster validation during training.",
    )
    return parser.parse_args()


def maybe_apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Install peft to use --use-lora") from exc

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        model.to(torch.device("cuda", local_rank))
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = maybe_apply_lora(model, args)

    train_dataset = JsonlDataset(args.train_jsonl)
    eval_dataset = JsonlDataset(args.eval_jsonl)
    if args.limit_train_samples is not None:
        train_dataset.rows = train_dataset.rows[: args.limit_train_samples]
    if args.limit_eval_samples is not None:
        eval_dataset.rows = eval_dataset.rows[: args.limit_eval_samples]
    collator = AFNextCollator(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
