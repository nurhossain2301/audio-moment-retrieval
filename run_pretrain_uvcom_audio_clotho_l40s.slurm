#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH -J "uvcom_clotho_pre"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mkhan@wpi.edu
#SBATCH --gres=gpu:1
#SBATCH -C L40S
#SBATCH -o logs/uvcom_audio_clotho_pretrain_%j.out
#SBATCH -e logs/uvcom_audio_clotho_pretrain_%j.err

set -euo pipefail

PROJECT_DIR="/home/mkhan/audio-moment-retrieval"
LIGHTHOUSE_DIR="$PROJECT_DIR/lighthouse"

cd "$PROJECT_DIR"
mkdir -p logs

module load cuda/12.6.3/5fe76nu

source "$PROJECT_DIR/venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================="
echo "UVCOM-audio Clotho-Moment pretraining"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Python: $(which python)"
nvidia-smi
echo ""

cd "$LIGHTHOUSE_DIR"

python training/train.py \
    --model uvcom_audio \
    --dataset clotho-moment \
    --feature clap

echo ""
echo "Finished at: $(date)"
echo "Results directory: $LIGHTHOUSE_DIR/results/uvcom_audio/clotho-moment/clap"
