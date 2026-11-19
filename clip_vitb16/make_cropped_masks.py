"""
STEP 0 - Put the target masks in the same 224x224 frame CLIP actually sees.

CLIPProcessor resizes the short side to 224 and centre-crops. Your masks from
prepare_exdark.py are full-resolution, so OAL computed against them would be
wrong, and some targets are partly (or entirely) cropped away.

Outputs (default split = Testing):
  data/exdark/processed/masks_clip224/testing_masks.npz
      image_id, target (N,224,224) uint8 soft mask*255, all_objects (same)
  data/exdark/processed/masks_clip224/testing_mask_stats.csv
      per-image crop area fractions, crop survival, flags
  data/exdark/processed/masks_clip224/qa_cropped_masks.png

Run (CPU is fine, ~5-10 min):
  python make_cropped_masks.py --split Testing
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exdark_common as C  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="Testing",
                   choices=["Training", "Validation", "Testing", "all"])
    p.add_argument("--crop-mode", default="auto",
                   choices=["auto", "floor", "round"])
    p.add_argument("--min-survival", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def detect_crop_mode(df, processor, n=40):
    """
    Compare our resize+crop of the RGB image with the processor's real output.
    Only images whose crop offset differs between the two rounding rules are
    informative.
    """
    ip = processor.image_processor
    mean = np.array(ip.image_mean, dtype=np.float32)[:, None, None]
    std = np.array(ip.image_std, dtype=np.float32)[:, None, None]

    errs = {"floor": [], "round": []}
    used = 0
    for _, row in df.iterrows():
        w, h = int(row["image_width"]), int(row["image_height"])
        nw, nh = C.resize_shape(w, h)
        if C.crop_offsets(nw, nh, mode="floor") == C.crop_offsets(nw, nh, mode="round"):
            continue
        pil = C.load_rgb(row["image_path"])
        ref = processor(images=[pil], return_tensors="pt")["pixel_values"][0].numpy()
        for mode in errs:
            mine = C.image_to_clip_frame(pil, mode).astype(np.float32) / 255.0
            mine = (mine.transpose(2, 0, 1) - mean) / std
            errs[mode].append(float(np.abs(mine - ref).mean()))
        used += 1
        if used >= n:
            break

    if used == 0:
        print("No image has an odd crop margin; rounding rule irrelevant -> floor")
        return "floor", {}

    summary = {m: float(np.mean(v)) for m, v in errs.items()}
    mode = min(summary, key=summary.get)
    print(f"Crop-mode check on {used} images (mean |diff| in normalised units): "
          f"{summary} -> using '{mode}'")
    return mode, summary


def main(argv=None, processor=None):
    args = parse_args(argv)
    C.CROPPED_MASK_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(C.IMAGE_CSV)
    if args.split != "all":
        df = df[df["split"] == args.split]
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Images: {len(df)}  (split={args.split})")

    # ---------------- crop rounding rule ----------------
    if args.crop_mode == "auto":
        if processor is None:
            from transformers import CLIPProcessor
            processor = CLIPProcessor.from_pretrained(C.MODEL_NAME)
        mode, mode_check = detect_crop_mode(df, processor)
    else:
        mode, mode_check = args.crop_mode, {}

    ids, tgt, allm, rows = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping masks"):
        t_full = np.asarray(Image.open(row["target_mask_path"]).convert("L"))
        a_full = np.asarray(Image.open(row["all_object_mask_path"]).convert("L"))

        t_crop, t_surv = C.mask_to_clip_frame(t_full, mode)
        a_crop, a_surv = C.mask_to_clip_frame(a_full, mode)

        ids.append(row["image_id"])
        tgt.append(np.round(t_crop * 255).astype(np.uint8))
        allm.append(np.round(a_crop * 255).astype(np.uint8))

        crop_area = float(t_crop.mean())
        no_target = (row["n_target_objects"] == 0) or (row["target_mask_pixels"] == 0)
        rows.append({
            "image_name": row["image_name"],
            "image_id": row["image_id"],
            "orig_target_area_fraction": row["target_area_fraction"],
            "crop_target_area_fraction": crop_area,
            "crop_all_object_area_fraction": float(a_crop.mean()),
            "target_crop_survival": t_surv,
            "all_object_crop_survival": a_surv,
            "no_target_box": bool(no_target),
            "target_cropped_out": bool((not no_target) and crop_area == 0),
            "low_survival": bool((not no_target) and t_surv < args.min_survival),
            "crop_mode": mode,
        })

    split_tag = args.split.lower()
    npz_path = C.CROPPED_MASK_DIR / f"{split_tag}_masks.npz"
    np.savez_compressed(npz_path, image_id=np.array(ids),
                        target=np.stack(tgt), all_objects=np.stack(allm))

    stats = pd.DataFrame(rows)
    stats_path = C.CROPPED_MASK_DIR / f"{split_tag}_mask_stats.csv"
    stats.to_csv(stats_path, index=False)

    summary = {
        "split": args.split,
        "n_images": len(stats),
        "crop_mode": mode,
        "crop_mode_check": mode_check,
        "no_target_box": int(stats["no_target_box"].sum()),
        "target_fully_cropped_out": int(stats["target_cropped_out"].sum()),
        f"survival_below_{args.min_survival}": int(stats["low_survival"].sum()),
        "median_survival": float(stats["target_crop_survival"].median()),
        "median_area_orig": float(stats["orig_target_area_fraction"].median()),
        "median_area_crop": float(stats["crop_target_area_fraction"].median()),
    }
    with open(C.CROPPED_MASK_DIR / f"{split_tag}_mask_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    # ---------------- QA figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qa = stats.sample(n=min(8, len(stats)), random_state=42)
    lookup = df.set_index("image_id")
    id_to_i = {k: i for i, k in enumerate(ids)}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for ax, (_, r) in zip(axes.ravel(), qa.iterrows()):
        pil = C.load_rgb(lookup.loc[r["image_id"], "image_path"])
        ax.imshow(C.image_to_clip_frame(pil, mode))
        m = tgt[id_to_i[r["image_id"]]] / 255.0
        ax.contour(m, levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(f"{lookup.loc[r['image_id'], 'class_name']}  "
                     f"surv={r['target_crop_survival']:.2f}\n"
                     f"area {r['orig_target_area_fraction']:.2f} -> "
                     f"{r['crop_target_area_fraction']:.2f}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(qa):]:
        ax.axis("off")
    fig.suptitle("Target masks in CLIP's 224x224 input frame (red = target box)")
    plt.tight_layout()
    plt.savefig(C.CROPPED_MASK_DIR / "qa_cropped_masks.png", dpi=150)
    plt.close()

    print("\nWrote:", npz_path, stats_path, sep="\n  ")


if __name__ == "__main__":
    main()
