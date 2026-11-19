#!/bin/bash
#SBATCH --job-name=clip-in9
#SBATCH --output=logs/clip-in9-%j.out
#SBATCH --error=logs/clip-in9-%j.err
#SBATCH --partition=htc
#SBATCH --qos=htc
#SBATCH --time=0-02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${DATA_ROOT:?Submit with DATA_ROOT=/path/to/extracted/data}"

module purge
module load mamba/latest
source activate clip-in9

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/cache/huggingface}"
mkdir -p logs results "$HF_HOME"

python evaluate_clip.py \
  --data-root "$DATA_ROOT" \
  --variants original,mixed_same,mixed_rand \
  --output-dir results \
  --batch-size 128 \
  --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
  --model-name RN50-quickgelu \
  --pretrained openai
