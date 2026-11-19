# CLIP on ImageNet-9: end-to-end preliminary experiment

This project evaluates the original OpenAI **CLIP RN50** checkpoint as a
zero-shot classifier on ImageNet-9. It loads `RN50` with `pretrained=openai`
through OpenCLIP. Start with three test variants:

- `original`: ordinary foreground/background combinations.
- `mixed_same`: foregrounds placed on backgrounds from the same class.
- `mixed_rand`: foregrounds placed on random backgrounds.
- `fg_mask`: foreground-mask test images used for an additional diagnostic.

The main robustness statistic is:

`BG-Gap = accuracy(mixed_same) - accuracy(mixed_rand)`

A larger gap indicates greater reliance on class-correlated backgrounds. The
official Backgrounds Challenge uses this same definition.

## 1. Connect to ASU Sol

Connect to the ASU Cisco VPN, then use either the Sol web portal or:

```bash
ssh YOUR_ASURITE@sol.asu.edu
```

Do not perform inference or package installation on a login node. First request
a short interactive compute session:

```bash
interactive -p lightwork -q public -t 0-02:00 -c 4 --mem=16G
```

If your access is through a class allocation, run `myaccounts` and replace
`-q public` with `-q class -A YOUR_CLASS_ACCOUNT` where required.

## 2. Put this project on Sol

On your Mac, unzip the downloaded project and transfer it:

```bash
scp -r clip_imagenet9 YOUR_ASURITE@sol.asu.edu:/scratch/YOUR_ASURITE/
```

Then, on Sol:

```bash
cd /scratch/$USER/clip_imagenet9
mkdir -p logs results cache/huggingface
```

## 3. Create the environment

Run this on the interactive compute session:

```bash
module load mamba/latest
mamba env create -f environment.yml
source activate clip-in9
python -c "import torch; print(torch.__version__)"
```

If the exact CUDA package combination is unavailable on Sol, create the
environment from `requirements.txt` inside a current ASU PyTorch environment.
Do not install into Mamba's base environment.

## 4. Download the official ImageNet-9 test data

Use the MadryLab Backgrounds Challenge release:

https://github.com/MadryLab/backgrounds_challenge/releases

Download the release asset named `backgrounds_challenge_data.tar.gz` into a
scratch data directory, not the Git repository. For example:

```bash
mkdir -p /scratch/$USER/datasets/imagenet9
cd /scratch/$USER/datasets/imagenet9
# Download backgrounds_challenge_data.tar.gz from the official release here.
tar -xzf backgrounds_challenge_data.tar.gz
find . -maxdepth 3 -type d | head -40
```

Find the directory that directly contains variant folders such as `original`,
`mixed_same`, `mixed_rand`, and `fg_mask`. That directory is `DATA_ROOT`. Each
variant should contain a `val` directory with nine class folders.

## 5. Cache CLIP RN50 once

Model downloading needs network access, while the batch job should be able to
run from cached files:

```bash
cd /scratch/$USER/clip_imagenet9
source activate clip-in9
export HF_HOME=$PWD/cache/huggingface
python - <<'PY'
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "RN50", pretrained="openai"
)
print("OpenAI CLIP RN50 cached successfully")
PY
```

## 6. Smoke-test the pipeline

Request one GPU for a short test:

```bash
interactive -p htc -q public -t 0-00:30 -G 1 -c 4 --mem=32G
cd /scratch/$USER/clip_imagenet9
module load mamba/latest
source activate clip-in9
export HF_HOME=$PWD/cache/huggingface
```

Run 10 images per class on all three variants:

```bash
python evaluate_clip.py \
  --data-root /scratch/$USER/datasets/imagenet9/PATH_TO_VARIANT_ROOT \
  --variants original,mixed_same,mixed_rand,fg_mask \
  --output-dir results_smoke \
  --max-images-per-class 10 \
  --batch-size 64 \
  --model-name RN50 \
  --pretrained openai
```

Confirm that `results_smoke/summary.csv`, `variant_accuracy.png`, and three
confusion-matrix plots were created. The smoke-test verifies the pipeline; do
not treat its small-sample accuracy as the final result.

## 7. Submit the full run

Edit `run_clip.slurm` if you use the class QoS/account. Then:

```bash
cd /scratch/$USER/clip_imagenet9
sbatch --export=ALL,DATA_ROOT=/scratch/$USER/datasets/imagenet9/PATH_TO_VARIANT_ROOT run_clip.slurm
squeue -u $USER
```

After it finishes:

```bash
cat logs/clip-in9-JOB_ID.out
column -s, -t results/summary.csv
```

If the job waits too long, leave `--gpus=1` unspecified by type so Slurm can
choose the first compatible GPU. CLIP RN50 does not need an A100.

## 8. What to put in the preliminary presentation

### Research question

Suggested wording:

> How robust are off-the-shelf vision-language models to changes in image
> background when performing zero-shot classification on ImageNet-9?

Explain that CLIP RN50 is being added as a convolutional zero-shot baseline for
comparison with LLaVA. The checkpoint uses OpenAI's pretrained weights and is
not fine-tuned on ImageNet-9.

### Working pipeline evidence

Show one slide with:

`ImageNet-9 image -> CLIP preprocessing -> image/text embeddings -> cosine similarity -> predicted one of nine classes -> metrics`

Include a screenshot of the completed log and the generated `summary.csv`.

### Preliminary result

Use `results/variant_accuracy.png` as the main plot. Report:

- Original accuracy and 95% bootstrap confidence interval.
- Mixed-Same accuracy.
- Mixed-Rand accuracy.
- BG-Gap in percentage points.
- Macro-F1 if class balance is uncertain.

Reasonable conclusion template:

> CLIP achieved ___% on Original, ___% on Mixed-Same, and ___% on
> Mixed-Rand. The ___-percentage-point BG-Gap indicates [limited/moderate/strong]
> sensitivity to background correlations. These are preliminary zero-shot
> results and do not yet establish that CLIP is more robust than LLaVA because
> both models must be evaluated on the same images and label set.

Do not describe softmax similarity values as calibrated probabilities.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| CLIP is sensitive to prompt wording | Freeze the five templates in the script; optionally report a later prompt ablation. |
| Broad IN-9 labels can be ambiguous | Use exactly the same nine labels for CLIP and LLaVA and inspect the confusion matrices. |
| LLaVA and CLIP outputs differ | Convert both to the same nine-class prediction schema and evaluate the same image list. |
| Small preliminary subset is noisy | Use a balanced subset for debugging, then run every official test image. |
| Background-only images may not contain the object | Treat those variants as background-reliance diagnostics, not ordinary object recognition. |

### Next steps

1. Run the full official test sets with a fixed seed and saved predictions.
2. Evaluate LLaVA on the exact same images and class names.
3. Put CLIP and LLaVA accuracies in one comparison plot.
4. Review high-confidence errors and per-class confusion matrices.
5. Optionally add `only_fg` and background-only variants.
6. Record model IDs, prompts, package versions, and Slurm job ID in the report.

## Output files

- `summary.csv`: accuracy, 95% confidence interval, macro-F1, and sample count.
- `variant_accuracy.png`: slide-ready accuracy comparison.
- `confusion_VARIANT.png`: normalized confusion matrices.
- `predictions_VARIANT.csv`: per-example labels, predictions, and similarity confidence.
- `run_metadata.json`: model ID, prompts, seed, device, and BG-Gap.
