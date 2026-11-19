#!/bin/bash
#SBATCH --job-name=clip-oal
#SBATCH --partition=htc
#SBATCH --qos=htc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/clip-oal-%j.out

set -euo pipefail

cd /scratch/$USER/clip_imagenet9
module load mamba/latest
source activate clip-in9
export HF_HOME="$PWD/cache/huggingface"

: "${DATA_ROOT:?Submit with sbatch --export=ALL,DATA_ROOT=/path/to/bg_challenge run_clip_oal.slurm}"

python evaluate_clip_oal.py \
  --data-root "$DATA_ROOT" \
  --variants original,mixed_same,mixed_rand \
  --mask-variant fg_mask \
  --output-dir results_oal_pilot \
  --images-per-class 10 \
  --model-name RN50-quickgelu \
  --pretrained openai \
  --cam-target predicted
