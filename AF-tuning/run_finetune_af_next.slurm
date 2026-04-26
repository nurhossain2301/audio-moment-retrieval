#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH -J "af_next_amr"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:4
#SBATCH -C L40S
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mkhan@wpi.edu

set -euo pipefail

cd /home/mkhan/audio-moment-retrieval
module load cuda/12.8.0/4fdo42o
module load ffmpeg/6.1.1/cup2q2r
source ./venv/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/npp/lib:/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/ffmpeg-6.1.1-cup2q2rgvykfd5goili4a7wwuex2zv33/lib:${LD_LIBRARY_PATH:-}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=WARN

echo "=========================================="
echo "AF-Next LoRA Fine-Tuning"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}, CUDA available: {torch.cuda.is_available()}')"
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    AF-tuning/scripts/finetune_af_next_lora.py \
    --model-path AF-tuning/models/audio-flamingo-next-hf \
    --train-jsonl AF-tuning/data/clotho_moment/train.jsonl \
    --eval-jsonl AF-tuning/data/clotho_moment/val.jsonl \
    --output-dir AF-tuning/outputs/af-next-clotho-lora \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --num-train-epochs 1 \
    --save-steps 500 \
    --eval-steps 500 \
    --logging-steps 10 \
    --limit-eval-samples 512

echo "End time: $(date)"
