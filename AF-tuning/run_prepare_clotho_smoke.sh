#!/bin/bash

set -euo pipefail

cd /home/mkhan/audio-moment-retrieval
if command -v module >/dev/null 2>&1; then
    module load ffmpeg/6.1.1/cup2q2r || true
fi
if [ -f AF-tuning/venv/bin/activate ]; then
    source AF-tuning/venv/bin/activate
else
    source ./venv/bin/activate
fi

python AF-tuning/scripts/prepare_clotho_moment.py \
    --splits train val \
    --extract-audio \
    --max-samples 64
