#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -J "castella_audio_dl"
#SBATCH -p short
#SBATCH -t 5:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mkhan@wpi.edu

set -euo pipefail

PROJECT_DIR="/home/mkhan/audio-moment-retrieval"
RAW_DIR="$PROJECT_DIR/raw_datasets/castella_audio"
DOWNLOADER="$PROJECT_DIR/CASTELLA-audio/script/download.py"

module load ffmpeg/6.1.1/cup2q2r

cd "$PROJECT_DIR"
source ./venv/bin/activate

mkdir -p "$RAW_DIR/download/audio"
cd "$RAW_DIR"

echo "=========================================="
echo "CASTELLA Raw Audio Download"
echo "=========================================="
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "yt-dlp: $(yt-dlp --version)"
echo "FFmpeg: $(ffmpeg -version | head -1)"
echo ""

for split in train val test; do
    echo "Downloading CASTELLA ${split} audio..."
    python "$DOWNLOADER" "$PROJECT_DIR/CASTELLA/json/en/${split}.json"
    echo "Finished ${split} at $(date)"
    echo ""
done

echo "Downloaded audio files:"
find "$RAW_DIR/download/audio" -maxdepth 1 -type f -name '*.wav' | wc -l

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
