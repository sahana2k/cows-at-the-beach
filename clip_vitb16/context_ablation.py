"""
Is the background HELPING or MISLEADING CLIP?  (answers "context is sometimes
legitimate evidence" - e.g. a striped dog is a dog because it is on a sofa)

Attribution (OAL) says WHERE the evidence is. This script tests whether the
context is NECESSARY, by intervention, on the same 224x224 input CLIP sees:

  full        original input
  obj_only    everything outside the target boxes replaced by a heavy blur
              of the image (keeps the dark look; no bright grey patches)
  obj_crop    target boxes cropped from the ORIGINAL image (square, +10%
              margin) and fed through CLIPProcessor - a natural-looking
              object-only view, used as a robustness check
  no_target   target boxes blurred out (what does the scene alone say?)
  no_objects  ALL annotated boxes blurred out (scenery only, no context objects)

Per image, with margin = logit(true class) - max logit(other classes):
  context_effect = margin(full) - margin(obj_only)   >0: context helped

Categories (full vs obj_only correctness):
  object_sufficient  correct with and without context
  context_needed     correct only WITH context       <- the striped-dog case
  context_misled     correct only WITHOUT context    <- harmful shortcut
  both_fail          wrong either way

  python context_ablation.py                       # all usable test images
  python context_ablation.py --method gradcam_L11  # + join with OAL table

Output: outputs/exdark/attribution/context/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exdark_common as ec  # noqa: E402
from deletion_test import gaussian_blur  # noqa: E402

CONDITIONS = ["full", "obj_only", "obj_crop", "no_target", "no_objects"]


def square_crop_box(boxes, w, h, margin=0.10):
    x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
    x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(side, w, h) if side > min(w, h) else side
    l = int(np.clip(cx - side / 2, 0, max(0, w - side)))
    t = int(np.clip(cy - side / 2, 0, max(0, h - side)))
    return l, t, int(min(w, l + side)), int(min(h, t + side))


def main(argv=None, model=None, processor=None, text_feats=None):
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--fill", default="blur", choices=["blur", "mean"])
    p.add_argument("--prompts", default="single", choices=["single", "ensemble"])
    p.add_argument("--method", default=None, help="join with attribution/<tag>/oal_table.csv")
    p.add_argument("--tag", default="full")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    device = ec.get_device()
    out_dir = ec.ATTR_DIR / "context"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ec.IMAGE_CSV)
    df = df[df["split"] == "Testing"]
    st = pd.read_csv(ec.CROPPED_MASK_DIR / "testing_mask_stats.csv")
    df = df.merge(st[["image_name", "no_target_box", "target_cropped_out",
                      "target_crop_survival"]], on="image_name")
    df = df[~df.no_target_box & ~df.target_cropped_out].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    mz = np.load(ec.CROPPED_MASK_DIR / "testing_masks.npz")
    mpos = {k: i for i, k in enumerate(mz["image_id"].astype(str))}
    objs = pd.read_csv(ec.OBJECT_CSV)
    objs["l"] = objs["object_class"].str.lower()
    og = dict(tuple(objs.groupby("image_name")))

    if model is None:
        model, processor = ec.load_clip(device)
    if text_feats is None:
        text_feats = ec.encode_text(model, processor, device, args.prompts)

    rows = []
    for s in tqdm(range(0, len(df), args.batch_size), desc="Context ablation"):
        b = df.iloc[s: s + args.batch_size]
        pils = [ec.load_rgb(pth) for pth in b["image_path"]]
        px = ec.preprocess(processor, pils, device)
        fill = (gaussian_blur(px) if args.fill == "blur"
                else px.mean(dim=(2, 3), keepdim=True).expand_as(px))
        ii = [mpos[str(i)] for i in b["image_id"]]
        mt = torch.as_tensor(mz["target"][ii] / 255.0, dtype=px.dtype, device=device)[:, None]
        ma = torch.as_tensor(mz["all_objects"][ii] / 255.0, dtype=px.dtype, device=device)[:, None]

        crops = []
        for pil, (_, r) in zip(pils, b.iterrows()):
            o = og[r["image_name"]]
            bx = o[o["l"] == r["class_name"].lower()][["x1", "y1", "x2", "y2"]].to_numpy()
            crops.append(pil.crop(square_crop_box(bx, *pil.size)))

        inputs = {
            "full": px,
            "obj_only": px * mt + fill * (1 - mt),
            "obj_crop": ec.preprocess(processor, crops, device),
            "no_target": px * (1 - mt) + fill * mt,
            "no_objects": px * (1 - ma) + fill * ma,
        }
        tgt = torch.tensor([ec.CLASS_TO_IDX[c] for c in b["class_name"]], device=device)
        res = {}
        for cond, x in inputs.items():
            lg = ec.class_logits(model, text_feats, x)
            t = lg.gather(1, tgt[:, None])[:, 0]
            other = lg.scatter(1, tgt[:, None], float("-inf")).max(1).values
            res[cond] = (lg.argmax(1).cpu().numpy(), (t - other).cpu().numpy())

        for j, (_, r) in enumerate(b.iterrows()):
            row = {"image_name": r["image_name"], "ground_truth": r["class_name"],
                   "lighting": r["lighting"], "target_crop_survival": r["target_crop_survival"]}
            for cond in CONDITIONS:
                row[f"pred_{cond}"] = ec.CLASS_NAMES[res[cond][0][j]]
                row[f"correct_{cond}"] = row[f"pred_{cond}"] == r["class_name"]
                row[f"margin_{cond}"] = float(res[cond][1][j])
            rows.append(row)

    d = pd.DataFrame(rows)
    d["context_effect"] = d["margin_full"] - d["margin_obj_only"]
    d["context_effect_crop"] = d["margin_full"] - d["margin_obj_crop"]
    cf, co = d["correct_full"], d["correct_obj_only"]
    d["category"] = np.select(
        [cf & co, cf & ~co, ~cf & co], ["object_sufficient", "context_needed", "context_misled"],
        "both_fail")
    d["category_crop"] = np.select(
        [cf & d.correct_obj_crop, cf & ~d.correct_obj_crop, ~cf & d.correct_obj_crop],
        ["object_sufficient", "context_needed", "context_misled"], "both_fail")

    # ---- join with OAL ----
    if args.method:
        o = pd.read_csv(ec.ATTR_DIR / args.tag / "oal_table.csv")
        o = o[["image_name", f"oal_{args.method}_pred", f"oal_{args.method}_target",
               f"attrin_allobj_{args.method}"]]
        d = d.merge(o, on="image_name", how="left")
        col = f"oal_{args.method}_target"
        d["focus"] = np.where(d[col] > 1, "object_focused", "context_focused")
        d["quadrant"] = d["focus"] + " / " + np.where(cf, "correct", "incorrect")

    d.to_csv(out_dir / "context_ablation.csv", index=False)

    # ---- summaries ----
    lines = []
    acc = {c: d[f"correct_{c}"].mean() for c in CONDITIONS}
    lines.append("Accuracy by condition (usable test images, n=%d):" % len(d))
    lines += [f"  {c:<11} {a * 100:5.1f}%" for c, a in acc.items()]
    lines.append("\nCategories (full vs obj_only blur):")
    lines.append(d["category"].value_counts(normalize=True).round(3).to_string())
    lines.append("\nCategories (full vs obj_crop) - robustness check:")
    lines.append(d["category_crop"].value_counts(normalize=True).round(3).to_string())
    lines.append(f"\nMedian context_effect (margin full - obj_only): "
                 f"{d.context_effect.median():.3f}   (>0 = context helps on average)")
    by_cls = d.groupby("ground_truth").agg(
        n=("category", "size"), acc_full=("correct_full", "mean"),
        acc_obj_only=("correct_obj_only", "mean"), acc_obj_crop=("correct_obj_crop", "mean"),
        acc_no_target=("correct_no_target", "mean"),
        context_needed=("category", lambda s: (s == "context_needed").mean()),
        context_misled=("category", lambda s: (s == "context_misled").mean()),
        median_context_effect=("context_effect", "median")).sort_values("acc_full")
    by_cls.to_csv(out_dir / "context_by_class.csv")
    lines.append("\nPer class:\n" + by_cls.round(3).to_string())
    d.groupby("lighting")[["correct_full", "correct_obj_only", "context_effect"]].agg(
        ["mean", "median"]).to_csv(out_dir / "context_by_lighting.csv")

    if args.method:
        col = f"oal_{args.method}_target"
        ok = d[[col, "context_effect"]].dropna()
        rho = stats.spearmanr(ok[col], ok["context_effect"])
        lines.append(f"\nSpearman(OAL_target, context_effect) = {rho.correlation:.3f} "
                     f"(p={rho.pvalue:.1e}); expect NEGATIVE if OAL is meaningful: "
                     "low object focus <-> removing context hurts more")
        q = d.groupby("quadrant").agg(n=("category", "size"),
                                      median_context_effect=("context_effect", "median"),
                                      context_needed=("category", lambda s: (s == "context_needed").mean()),
                                      context_misled=("category", lambda s: (s == "context_misled").mean()))
        q.to_csv(out_dir / "quadrants.csv")
        lines.append("\nAttribution x outcome quadrants:\n" + q.round(3).to_string())

    txt = "\n".join(lines)
    (out_dir / "context_summary.txt").write_text(txt + "\n")
    print(txt)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["object_sufficient", "context_needed", "context_misled", "both_fail"]
    ct = pd.crosstab(d["ground_truth"], d["category"], normalize="index").reindex(columns=order, fill_value=0)
    ct = ct.loc[by_cls.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    left = np.zeros(len(ct))
    for c in order:
        ax.barh(ct.index, ct[c], left=left, label=c.replace("_", " "))
        left += ct[c].values
    ax.set(xlabel="Fraction of test images", xlim=(0, 1),
           title="Does the background help or mislead CLIP ViT-B/16 on ExDark?")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_context_categories.png", dpi=200)
    plt.close()
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
