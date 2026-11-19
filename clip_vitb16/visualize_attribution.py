"""
STEP 2 - Look at the maps.

Writes, for a run tag:
  overlays/compare_methods_pXX.png   rows = images, cols = methods (pred-class map)
  overlays/<method>_pXX.png          rows = images, cols = input+box | pred map | target map

  python visualize_attribution.py --tag pilot
  python visualize_attribution.py --tag full --images 2015_00001.jpg 2015_00410.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exdark_common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def norm01(h):
    h = h - h.min()
    return h / h.max() if h.max() > 0 else h


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--images", nargs="+", default=None, help="image_name list")
    p.add_argument("--rows-per-page", type=int, default=8)
    p.add_argument("--max-images", type=int, default=64)
    args = p.parse_args(argv)

    run_dir = C.ATTR_DIR / args.tag
    out_dir = run_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(run_dir / "maps.npz")
    names = z["image_name"].astype(str)
    methods = args.methods or sorted({k.split("__")[0] for k in z.files if "__" in k})

    oal = pd.read_csv(run_dir / "oal_table.csv").set_index("image_name")
    meta = pd.read_csv(C.IMAGE_CSV).set_index("image_name")
    mz = np.load(C.CROPPED_MASK_DIR / "testing_masks.npz")
    mpos = {k: i for i, k in enumerate(mz["image_id"].astype(str))}
    mode = oal["crop_mode"].iloc[0]

    idx = np.arange(len(names))
    if args.images:
        want = set(args.images)
        idx = np.array([i for i in idx if names[i] in want])
    else:
        # incorrect first: they are the interesting ones
        idx = sorted(idx, key=lambda i: (bool(oal.loc[names[i], "correct"]), names[i]))
    idx = list(idx)[: args.max_images]

    cache = {}

    def crop_and_mask(i):
        if i not in cache:
            n = names[i]
            img = C.image_to_clip_frame(C.load_rgb(meta.loc[n, "image_path"]), mode)
            msk = mz["target"][mpos[oal.loc[n, "image_id"]]] / 255.0
            cache[i] = (img, msk)
        return cache[i]

    def draw(ax, img, msk, heat=None, title=""):
        ax.imshow(img)
        if heat is not None:
            ax.imshow(norm01(heat), cmap="jet", alpha=0.45)
        if msk.max() > 0:
            ax.contour(msk, levels=[0.5], colors="white", linewidths=1.2)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    pages = [idx[k: k + args.rows_per_page] for k in range(0, len(idx), args.rows_per_page)]

    # ---- method comparison pages ----
    for pg, rows in enumerate(pages):
        fig, axes = plt.subplots(len(rows), len(methods) + 1,
                                 figsize=(2.2 * (len(methods) + 1), 2.4 * len(rows)),
                                 squeeze=False)
        for r, i in enumerate(rows):
            n = names[i]
            img, msk = crop_and_mask(i)
            row = oal.loc[n]
            mark = "OK" if row["correct"] else "WRONG"
            draw(axes[r, 0], img, msk,
                 title=f"{n}\nGT {row['ground_truth']} | CLIP {row['prediction']} {mark}")
            for c, m in enumerate(methods, start=1):
                h = C.upsample(z[f"{m}__pred"][i: i + 1])[0]
                draw(axes[r, c], img, msk, h,
                     f"{m}\nOAL={row[f'oal_{m}_pred']:.2f}" if r == 0
                     else f"OAL={row[f'oal_{m}_pred']:.2f}")
        fig.suptitle(f"Predicted-class maps by method (white = target box) - page {pg + 1}")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(out_dir / f"compare_methods_p{pg + 1:02d}.png", dpi=130)
        plt.close()

    # ---- per-method pred vs target pages ----
    for m in methods:
        for pg, rows in enumerate(pages):
            fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.5 * len(rows)),
                                     squeeze=False)
            for r, i in enumerate(rows):
                n = names[i]
                img, msk = crop_and_mask(i)
                row = oal.loc[n]
                hp = C.upsample(z[f"{m}__pred"][i: i + 1])[0]
                ht = C.upsample(z[f"{m}__target"][i: i + 1])[0]
                draw(axes[r, 0], img, msk,
                     title=f"{n}  {row['lighting']}\nGT {row['ground_truth']} | "
                           f"area {row['crop_target_area_fraction']:.2f}")
                draw(axes[r, 1], img, msk, hp,
                     f"map for '{row['prediction']}' (pred)\nOAL={row[f'oal_{m}_pred']:.2f}")
                draw(axes[r, 2], img, msk, ht,
                     f"map for '{row['ground_truth']}' (target)\nOAL={row[f'oal_{m}_target']:.2f}")
            fig.suptitle(f"{m} - page {pg + 1}")
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.savefig(out_dir / f"{m}_p{pg + 1:02d}.png", dpi=130)
            plt.close()

    print("Wrote overlays to", out_dir)


if __name__ == "__main__":
    main()
