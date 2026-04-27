#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH -J "af_next_eval"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
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

echo "=========================================="
echo "AF-Next LoRA Clotho Test Evaluation"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}, CUDA available: {torch.cuda.is_available()}')"
echo ""

python AF-tuning/scripts/evaluate_af_next_lora.py \
    --model-path AF-tuning/models/audio-flamingo-next-hf \
    --adapter-path AF-tuning/outputs/af-next-clotho-lora \
    --test-jsonl AF-tuning/data/clotho_moment/test.jsonl \
    --output-jsonl AF-tuning/outputs/af-next-clotho-lora/test_predictions.jsonl \
    --metrics-json AF-tuning/outputs/af-next-clotho-lora/test_metrics.json \
    --max-new-tokens 128

echo "End time: $(date)"
