#!/bin/bash
# Full test split (2,563 images): attribution -> OAL -> deletion check on a
# 600-image sample -> RQ1 analysis with the method you chose from the pilot.
#
#   sbatch --export=ALL,METHOD=gradcam_L11 scripts/exdark/run_attr_full.sh

#SBATCH --job-name=exdark_attr_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=./outputs/exdark/slurm_attr_full_%j.out
#SBATCH --error=./outputs/exdark/slurm_attr_full_%j.err

set -e

METHOD="${METHOD:-gradcam_L11}"

PROJECT_DIR="$HOME/cse598_vlm_project"
PYTHON="/scratch/sholkal/envs/cse598-vlm/bin/python"
S="${PROJECT_DIR}/scripts/exdark"

export PIP_CACHE_DIR=/scratch/sholkal/pip-cache
export HF_HOME=/scratch/sholkal/huggingface
export HF_HUB_CACHE=/scratch/sholkal/huggingface/hub

cd "${PROJECT_DIR}"
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date) | METHOD=${METHOD}"

if [ ! -f data/exdark/processed/masks_clip224/testing_masks.npz ]; then
    "${PYTHON}" "${S}/make_cropped_masks.py" --split Testing
fi

"${PYTHON}" "${S}/run_attribution.py" --tag full --batch-size 16
"${PYTHON}" "${S}/compute_oal.py" --tag full
"${PYTHON}" "${S}/deletion_test.py" --tag full --sample 600 --batch-size 32
"${PYTHON}" "${S}/deletion_test.py" --tag full --sample 600 --batch-size 32 --fill mean
"${PYTHON}" "${S}/analyze_oal.py" --tag full --method "${METHOD}"
"${PYTHON}" "${S}/context_ablation.py" --tag full --method "${METHOD}" --batch-size 32

echo "Done at $(date)"
