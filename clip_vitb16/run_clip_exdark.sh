#!/bin/bash

#SBATCH --job-name=clip_exdark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=./outputs/exdark/slurm_%j.out
#SBATCH --error=./outputs/exdark/slurm_%j.err


set -e


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR="$HOME/cse598_vlm_project"

PYTHON="/scratch/sholkal/envs/cse598-vlm/bin/python"

SCRIPT_PATH="${PROJECT_DIR}/scripts/exdark/run_clip_exdark.py"

OUTPUT_DIR="${PROJECT_DIR}/outputs/exdark"


# ============================================================
# CACHE LOCATIONS
# ============================================================

export PIP_CACHE_DIR=/scratch/sholkal/pip-cache
export HF_HOME=/scratch/sholkal/huggingface
export HF_HUB_CACHE=/scratch/sholkal/huggingface/hub


mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"


# ============================================================
# JOB INFO
# ============================================================

echo "============================================================"
echo "CLIP ViT-B/16 ExDark Evaluation"
echo "============================================================"

echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Start time: $(date)"
echo ""

echo "Python:"
echo "${PYTHON}"

"${PYTHON}" --version


# ============================================================
# VERIFY ENVIRONMENT
# ============================================================

echo ""
echo "============================================================"
echo "ENVIRONMENT CHECK"
echo "============================================================"

"${PYTHON}" - <<'PY'
import sys
import torch
import transformers
import pandas
from PIL import Image

print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Pandas:", pandas.__version__)
print("Environment imports: OK")
PY


# ============================================================
# GPU CHECK
# ============================================================

echo ""
echo "============================================================"
echo "GPU CHECK"
echo "============================================================"

nvidia-smi

echo ""

"${PYTHON}" - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in this GPU job."
    )

print("GPU:", torch.cuda.get_device_name(0))

props = torch.cuda.get_device_properties(0)

print(
    "GPU memory:",
    round(props.total_memory / (1024 ** 3), 2),
    "GB"
)
PY


# ============================================================
# DATA CHECK
# ============================================================

echo ""
echo "============================================================"
echo "DATA CHECK"
echo "============================================================"

METADATA="${PROJECT_DIR}/data/exdark/processed/metadata/exdark_images.csv"

if [ ! -f "${METADATA}" ]; then
    echo "ERROR: Metadata not found:"
    echo "${METADATA}"
    exit 1
fi

echo "Metadata found:"
echo "${METADATA}"

echo ""
echo "Physical ExDark location:"
readlink -f "${PROJECT_DIR}/data/exdark"


# ============================================================
# CLIP INFERENCE
# ============================================================

echo ""
echo "============================================================"
echo "STARTING CLIP INFERENCE"
echo "============================================================"

"${PYTHON}" "${SCRIPT_PATH}" \
    --split Testing \
    --batch-size 64 \
    --output "${OUTPUT_DIR}/clip_vitb16_testing_predictions.csv"


# ============================================================
# COMPLETE
# ============================================================

echo ""
echo "============================================================"
echo "CLIP EXDARK EVALUATION COMPLETE"
echo "============================================================"

echo "End time: $(date)"

echo ""
echo "Predictions:"
echo "${OUTPUT_DIR}/clip_vitb16_testing_predictions.csv"