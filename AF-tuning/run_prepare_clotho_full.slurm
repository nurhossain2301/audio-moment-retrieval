#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH -J "af_clotho_prep"
#SBATCH -p short
#SBATCH -t 5:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mkhan@wpi.edu

set -euo pipefail

cd /home/mkhan/audio-moment-retrieval

module load ffmpeg/6.1.1/cup2q2r

if [ -f AF-tuning/venv/bin/activate ]; then
    source AF-tuning/venv/bin/activate
else
    source ./venv/bin/activate
fi

echo "=========================================="
echo "Preparing full Clotho-Moment for AF-Next"
echo "=========================================="
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "FFmpeg: $(ffmpeg -version | head -1)"
echo ""

python AF-tuning/scripts/prepare_clotho_moment.py \
    --splits train val test \
    --extract-audio

echo ""
echo "Prepared record counts:"
wc -l AF-tuning/data/clotho_moment/train.jsonl \
      AF-tuning/data/clotho_moment/val.jsonl \
      AF-tuning/data/clotho_moment/test.jsonl

echo ""
echo "Extracted WAV count:"
find AF-tuning/data/clotho_moment/audio -type f -name '*.wav' | wc -l

echo "End time: $(date)"
