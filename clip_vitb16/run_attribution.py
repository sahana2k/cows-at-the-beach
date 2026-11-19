"""
STEP 1b / 4 - Attribution maps for CLIP ViT-B/16 on ExDark.

For every image, computes maps for BOTH
  * the predicted class  ("__pred")  -> comparable with the RN50 branch
  * the ExDark target    ("__target") -> "how much evidence for the right
                                          label sits on the object"
with several methods at once:
  gradcam_L11, gradcam_L10, gradcam_L8, ...  Grad-CAM on block-L patch tokens
  chefer_full                               Chefer et al. 2021, all blocks
  chefer_last                               Chefer, last block only
                                            (default in their CLIP notebook)

Saves raw 14x14 maps so OAL can be recomputed without the GPU:
  outputs/exdark/attribution/<tag>/maps.npz
  outputs/exdark/attribution/<tag>/run_predictions.csv

Pilot:  python run_attribution.py --tag pilot --manifest outputs/exdark/attribution/pilot_manifest.csv
Full:   python run_attribution.py --tag full
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exdark_common as C  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="e.g. pilot, full")
    p.add_argument("--manifest", default=None,
                   help="CSV with an image_name column (optional)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradcam-layers", type=int, nargs="+", default=[11, 10, 8, 6])
    p.add_argument("--prompts", default="single", choices=["single", "ensemble"])
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None, model=None, processor=None, text_feats=None):
    args = parse_args(argv)
    device = C.get_device()
    out_dir = C.ATTR_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(C.IMAGE_CSV)
    df = df[df["split"] == "Testing"].reset_index(drop=True)
    if args.manifest:
        keep = set(pd.read_csv(args.manifest)["image_name"])
        df = df[df["image_name"].isin(keep)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Device {device} | images {len(df)} | methods "
          f"{C.method_names(args.gradcam_layers)}")

    if model is None:
        model, processor = C.load_clip(device)
    if text_feats is None:
        text_feats = C.encode_text(model, processor, device, args.prompts)

    maps = {}
    logits_all, pred_idx_all, tgt_idx_all = [], [], []
    t0 = time.time()

    for start in tqdm(range(0, len(df), args.batch_size), desc="Attribution"):
        b = df.iloc[start: start + args.batch_size]
        px = C.preprocess(processor, [C.load_rgb(p) for p in b["image_path"]], device)

        if start == 0:
            err = C.check_forward_matches(model, px)
            print(f"Manual ViT forward matches HF get_image_features (max err {err:.1e})")

        logits = C.class_logits(model, text_feats, px)
        pred_idx = logits.argmax(1)
        tgt_idx = torch.tensor([C.CLASS_TO_IDX[c] for c in b["class_name"]],
                               device=device)

        for which, idx in (("pred", pred_idx), ("target", tgt_idx)):
            m, _ = C.compute_attributions(model, text_feats, px, idx,
                                          args.gradcam_layers)
            for k, v in m.items():
                maps.setdefault(f"{k}__{which}", []).append(v)

        logits_all.append(logits.cpu().numpy())
        pred_idx_all.append(pred_idx.cpu().numpy())
        tgt_idx_all.append(tgt_idx.cpu().numpy())

    logits_all = np.concatenate(logits_all)
    pred_idx_all = np.concatenate(pred_idx_all)
    tgt_idx_all = np.concatenate(tgt_idx_all)

    np.savez_compressed(
        out_dir / "maps.npz",
        image_id=df["image_id"].to_numpy().astype(str),
        image_name=df["image_name"].to_numpy().astype(str),
        pred_idx=pred_idx_all, target_idx=tgt_idx_all, logits=logits_all,
        **{k: np.concatenate(v).astype(np.float32) for k, v in maps.items()},
    )

    run = pd.DataFrame({
        "image_name": df["image_name"],
        "image_id": df["image_id"],
        "ground_truth": df["class_name"],
        "prediction_run": [C.CLASS_NAMES[i] for i in pred_idx_all],
    })
    run["correct_run"] = run["ground_truth"] == run["prediction_run"]

    # Compare with the baseline predictions (should be ~100% identical when
    # --prompts single).
    agree = None
    if C.PREDICTIONS_CSV.exists():
        base = pd.read_csv(C.PREDICTIONS_CSV)[["image_name", "prediction"]]
        run = run.merge(base, on="image_name", how="left")
        agree = float((run["prediction"] == run["prediction_run"]).mean())
        print(f"Agreement with baseline predictions CSV: {agree:.4f}")
    run.to_csv(out_dir / "run_predictions.csv", index=False)

    meta = {
        "tag": args.tag, "n_images": len(df), "prompts": args.prompts,
        "gradcam_layers": args.gradcam_layers,
        "methods": C.method_names(args.gradcam_layers),
        "accuracy_run": float(run["correct_run"].mean()),
        "agreement_with_baseline": agree,
        "seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
