#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$PROJECT_DIR/raw_datasets/castella_audio/download/audio"
COOKIES_ARG=()

if [ "${1:-}" != "" ]; then
    COOKIES_ARG=(--cookies "$1")
fi

source "$PROJECT_DIR/venv/bin/activate"
mkdir -p "$RAW_DIR"

for split in train val test; do
    echo "Downloading CASTELLA ${split} audio to $RAW_DIR"
    python "$PROJECT_DIR/CASTELLA-audio/script/download.py" \
        "$PROJECT_DIR/CASTELLA/json/en/${split}.json" \
        --output-dir "$RAW_DIR" \
        --stream-ranges \
        "${COOKIES_ARG[@]}"
done

echo "Downloaded WAV files:"
find "$RAW_DIR" -maxdepth 1 -type f -name '*.wav' | wc -l
