# CLIP ViT-B/16 × ExDark: attribution → OAL → RQ1

These scripts sit next to the ones you already have in
`~/cse598_vlm_project/scripts/exdark/`. They read your existing outputs:
`exdark_images.csv`, `exdark_objects.csv`, the full-resolution masks, and
`clip_vitb16_testing_predictions.csv`. They never modify those files.

| Script | Step | GPU? | What it does |
|---|---|---|---|
| `exdark_common.py` | – | – | Shared paths, prompts, model loading, ViT forward, Grad-CAM, Chefer, crop geometry, OAL |
| `make_cropped_masks.py` | 0 | no | Applies CLIP's resize + centre crop to the masks; computes crop survival and flags |
| `make_pilot_manifest.py` | 1a | no | Picks a fixed, stratified pilot set of about 60 images |
| `run_attribution.py` | 1b / 4 | yes | Computes 14×14 maps for each method, for both the predicted and the target class |
| `compute_oal.py` | 2 / 4 | no | Builds the OAL table: one row per image, columns for every method |
| `visualize_attribution.py` | 2 | no | Draws overlays: a method-comparison grid, then predicted vs target maps |
| `deletion_test.py` | 3 | yes | Faithfulness check: attributed patches vs random patches |
| `context_ablation.py` | 7 | yes | Context intervention: is the background helping or misleading? Removes the background, crops the object and removes objects, then joins the results to OAL |
| `analyze_oal.py` | 6 | no | RQ1 statistics, size control, logistic regression, per-class results, error types, figures |
| `run_attr_pilot.sh` / `run_attr_full.sh` | – | – | sbatch wrappers using the same header and environment as `run_clip_exdark.sh` |

Outputs go to `outputs/exdark/attribution/<tag>/` and
`data/exdark/processed/masks_clip224/`.

## One-time setup on Sol
```bash
cd ~/cse598_vlm_project
# copy the new files into scripts/exdark/ (scp / Sol OnDemand upload), then:
/scratch/sholkal/envs/cse598-vlm/bin/pip install statsmodels scipy scikit-learn
chmod +x scripts/exdark/*.sh
```

## Order
```bash
sbatch scripts/exdark/run_attr_pilot.sh                          # about 10 minutes
# read the pilot results, choose a method (see below), then:
sbatch --export=ALL,METHOD=<chosen_method> scripts/exdark/run_attr_full.sh
```

To change only the analysis later, use CPU-only and rerun from the saved maps:
`python scripts/exdark/analyze_oal.py --tag full --method chefer_full`

## Choosing the method
Use `attribution/pilot/deletion/deletion_summary.csv`. Pick the method with the
largest `mean_gain_over_random` that also beats random on most images. Then
confirm with `attribution/full/deletion/deletion_summary.csv` (600-image
sample). Do not choose based on which overlays look nicest.

Grad-CAM must never use block 12. CLIP's output reads only the CLS token of the
last block, so gradients on its patch tokens are exactly zero. The code
enforces L ≤ 11.
