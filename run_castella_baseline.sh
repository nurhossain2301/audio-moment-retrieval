#!/bin/bash

#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=256g
#SBATCH -J "castella"
#SBATCH -p short
#SBATCH -t 5:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mkhan@wpi.edu
#SBATCH --gres=gpu:1
#SBATCH -C L40S

# Load CUDA module
module load cuda/12.6.3/5fe76nu

# Navigate to project directory
cd /home/mkhan/audio-moment-retrieval

# Activate virtual environment
source ./venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CHECKPOINT="lighthouse/gradio_demo/weights/clap_qd_detr_finetuned.ckpt"
PRED_JSON="results/castella_baseline_predictions.json"
PRED_JSONL="results/castella_baseline_predictions.jsonl"
METRICS_JSON="results/castella_evaluation_metrics.json"
OFFICIAL_METRICS_JSON="results/castella_official_evaluation_metrics.json"

echo "=========================================="
echo "CASTELLA Baseline Inference & Evaluation"
echo "=========================================="
echo "Start time: $(date)"
echo "CUDA version: $(nvidia-smi | grep -i cuda | head -1)"
echo ""

# Step 1: Generate baseline predictions
echo "Step 1: Generating baseline predictions..."
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Missing official CASTELLA baseline checkpoint: $CHECKPOINT"
    echo "Download clap_qd_detr_finetuned.ckpt from https://zenodo.org/records/17422909 first."
    exit 1
fi

python generate_castella_baseline_results.py \
    --checkpoint "$CHECKPOINT" \
    --output "$PRED_JSON" \
    --jsonl-output "$PRED_JSONL"
STEP1_EXIT=$?

if [ $STEP1_EXIT -ne 0 ]; then
    echo "ERROR: Baseline prediction generation failed with exit code $STEP1_EXIT"
    exit $STEP1_EXIT
fi

echo ""
echo "Baseline predictions saved."
echo ""

# Step 2: Evaluate predictions
echo "Step 2: Evaluating predictions..."
python evaluate_castella_predictions.py \
    --predictions "$PRED_JSON" \
    --output "$METRICS_JSON" \
    --top-ks 1 5 10 \
    --iou-thresholds 0.3 0.5 0.7
STEP2_EXIT=$?

if [ $STEP2_EXIT -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $STEP2_EXIT"
    exit $STEP2_EXIT
fi

echo ""
echo "Evaluation complete."
echo "Metrics saved to $METRICS_JSON"
echo ""

# Step 3: Evaluate with the Lighthouse standalone evaluator used by the repo.
echo "Step 3: Running Lighthouse official-style evaluator..."
(
    cd lighthouse/training
    PYTHONPATH=. python -m standalone_eval.eval \
        --submission_path "../../$PRED_JSONL" \
        --gt_path "../data/castella/castella_test_release.jsonl" \
        --save_path "../../$OFFICIAL_METRICS_JSON"
)
STEP3_EXIT=$?

if [ $STEP3_EXIT -ne 0 ]; then
    echo "ERROR: Official-style evaluation failed with exit code $STEP3_EXIT"
    exit $STEP3_EXIT
fi

echo ""
echo "Official-style metrics saved to $OFFICIAL_METRICS_JSON"
echo ""

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit 0
