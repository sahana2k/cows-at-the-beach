#!/bin/bash
# Pilot: cropped masks -> pilot manifest -> attribution (~60 imgs) ->
#        deletion test -> OAL -> overlays.      Submit:  sbatch scripts/exdark/run_attr_pilot.sh

#SBATCH --job-name=exdark_attr_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=./outputs/exdark/slurm_attr_pilot_%j.out
#SBATCH --error=./outputs/exdark/slurm_attr_pilot_%j.err

set -e

PROJECT_DIR="$HOME/cse598_vlm_project"
PYTHON="/scratch/sholkal/envs/cse598-vlm/bin/python"
S="${PROJECT_DIR}/scripts/exdark"

export PIP_CACHE_DIR=/scratch/sholkal/pip-cache
export HF_HOME=/scratch/sholkal/huggingface
export HF_HUB_CACHE=/scratch/sholkal/huggingface/hub

cd "${PROJECT_DIR}"
mkdir -p outputs/exdark/attribution
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Step 0: masks in CLIP's 224x224 frame (skipped if already done)
if [ ! -f data/exdark/processed/masks_clip224/testing_masks.npz ]; then
    "${PYTHON}" "${S}/make_cropped_masks.py" --split Testing
fi

# Step 1: pilot set + attribution maps
"${PYTHON}" "${S}/make_pilot_manifest.py"
"${PYTHON}" "${S}/run_attribution.py" --tag pilot \
    --manifest outputs/exdark/attribution/pilot_manifest.csv --batch-size 16

# Step 2-3: OAL, overlays, deletion test
"${PYTHON}" "${S}/compute_oal.py" --tag pilot
"${PYTHON}" "${S}/visualize_attribution.py" --tag pilot
"${PYTHON}" "${S}/deletion_test.py" --tag pilot --batch-size 32

echo "Done at $(date)"
