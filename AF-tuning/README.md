# AF-Next Tuning for Audio Moment Retrieval

This folder contains a clean, isolated experiment scaffold for adapting
Audio Flamingo Next to audio moment retrieval.

The starting point is `nvidia/audio-flamingo-next-hf`, an 8B BF16
audio-text-to-text model. The model is released for non-commercial research.

## Layout

- `models/audio-flamingo-next-hf/`: local Hugging Face snapshot.
- `data/clotho_moment/`: prepared Clotho-Moment instruction records and audio.
- `outputs/`: fine-tuning checkpoints and logs.
- `scripts/download_af_next.py`: downloads the AF-Next model snapshot.
- `scripts/prepare_clotho_moment.py`: builds JSONL training records and can extract/resample audio.
- `scripts/finetune_af_next_lora.py`: LoRA/QLoRA-style supervised fine-tuning entrypoint.

## Download AF-Next

Create the isolated AF-Next environment first. Do not install the AF-Next
Transformers branch into the main Lighthouse venv because it conflicts with
Lighthouse's pinned `transformers<=4.51.3`.

```bash
bash AF-tuning/setup_env.sh
source AF-tuning/venv/bin/activate
```

```bash
python AF-tuning/scripts/download_af_next.py
```

## Prepare Clotho-Moment

Build manifests only:

```bash
python AF-tuning/scripts/prepare_clotho_moment.py --splits train val
```

Extract and resample audio needed by the manifests:

```bash
python AF-tuning/scripts/prepare_clotho_moment.py --splits train val --extract-audio
```

Use `--max-samples` for a small smoke-test set before full extraction.

## Fine-Tune

```bash
python AF-tuning/scripts/finetune_af_next_lora.py \
  --model-path AF-tuning/models/audio-flamingo-next-hf \
  --train-jsonl AF-tuning/data/clotho_moment/train.jsonl \
  --eval-jsonl AF-tuning/data/clotho_moment/val.jsonl \
  --output-dir AF-tuning/outputs/af-next-clotho-lora
```

The fine-tuning script expects a GPU with enough memory for an 8B multimodal
model. Start with a small prepared subset and `--max-steps 10` before launching
a long run.
