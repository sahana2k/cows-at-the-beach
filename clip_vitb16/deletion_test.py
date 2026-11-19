"""
STEP 3 - Faithfulness check: do the maps point at evidence CLIP actually uses?

For each method: rank the 196 patches by attribution, replace the top-k%
patches, and measure the drop in the explained class's logit. Compare against
replacing random patches (several seeds). A method is only usable if its drop
is clearly larger than random.

Fill choice matters on ExDark: grey / ImageNet-mean patches BRIGHTEN a dark
image (itself a distribution shift). Default fill = heavily blurred copy of the
same image; --fill mean uses the image's own mean colour instead.

Reads the maps saved by run_attribution.py (no new attribution needed).

  python deletion_test.py --tag pilot
  python deletion_test.py --tag full --sample 600     # more stable numbers

Outputs in outputs/exdark/attribution/<tag>/deletion/:
  deletion_long.csv, deletion_summary.csv, deletion_curves.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exdark_common as C  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--which", default="pred", choices=["pred", "target"])
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.05, 0.10, 0.20, 0.40])
    p.add_argument("--random-seeds", type=int, default=5)
    p.add_argument("--fill", default="blur", choices=["blur", "mean"])
    p.add_argument("--sample", type=int, default=None,
                   help="random subset of images (stratified by correctness)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--prompts", default="single", choices=["single", "ensemble"])
    return p.parse_args(argv)


def gaussian_blur(x, sigma=8.0):
    k = int(6 * sigma) | 1
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    Cn = x.shape[1]
    x = F.pad(x, (k // 2,) * 4, mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(Cn, 1, 1, k), groups=Cn)
    x = F.conv2d(x, g.view(1, 1, k, 1).expand(Cn, 1, k, 1), groups=Cn)
    return x


def patch_mask(order, n):
    """order: (B,196) patch indices by decreasing score -> (B,1,224,224)."""
    B = order.shape[0]
    m = torch.zeros(B, C.GRID * C.GRID, device=order.device)
    m.scatter_(1, order[:, :n], 1.0)
    m = m.view(B, 1, C.GRID, C.GRID)
    return m.repeat_interleave(C.PATCH, 2).repeat_interleave(C.PATCH, 3)


def main(argv=None, model=None, processor=None, text_feats=None):
    args = parse_args(argv)
    device = C.get_device()
    run_dir = C.ATTR_DIR / args.tag
    out_dir = run_dir / ("deletion" if args.fill == "blur" else f"deletion_{args.fill}")
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(run_dir / "maps.npz")
    methods = sorted({k.split("__")[0] for k in z.files if "__" in k})
    names = z["image_name"]
    cls_idx = z["pred_idx"] if args.which == "pred" else z["target_idx"]
    correct = z["pred_idx"] == z["target_idx"]

    sel = np.arange(len(names))
    if args.sample and args.sample < len(sel):
        rng = np.random.default_rng(0)
        frac = args.sample / len(sel)
        sel = np.concatenate([
            rng.choice(np.where(correct == c)[0],
                       size=max(1, int(round(frac * (correct == c).sum()))),
                       replace=False)
            for c in (True, False) if (correct == c).any()
        ])
        sel.sort()

    meta = pd.read_csv(C.IMAGE_CSV).set_index("image_name")
    if model is None:
        model, processor = C.load_clip(device)
    if text_feats is None:
        text_feats = C.encode_text(model, processor, device, args.prompts)

    n_patches = C.GRID * C.GRID
    rows = []

    for s in tqdm(range(0, len(sel), args.batch_size), desc="Deletion"):
        ib = sel[s: s + args.batch_size]
        px = C.preprocess(processor,
                          [C.load_rgb(meta.loc[n, "image_path"]) for n in names[ib]],
                          device)
        fill = (gaussian_blur(px) if args.fill == "blur"
                else px.mean(dim=(2, 3), keepdim=True).expand_as(px))
        ci = torch.as_tensor(cls_idx[ib], device=device).long()

        def score(x):
            lg = C.class_logits(model, text_feats, x)
            return (lg.gather(1, ci[:, None])[:, 0].cpu().numpy(),
                    torch.softmax(lg, 1).gather(1, ci[:, None])[:, 0].cpu().numpy())

        base_logit, base_prob = score(px)

        orders = {}
        for m in methods:
            a = torch.as_tensor(z[f"{m}__{args.which}"][ib], device=device)
            a = a.view(len(ib), -1) + 1e-9 * torch.rand(len(ib), n_patches,
                                                       device=device)  # break ties
            orders[m] = a.argsort(dim=1, descending=True)
        g = torch.Generator(device="cpu").manual_seed(1234 + s)
        for r in range(args.random_seeds):
            orders[f"random_s{r}"] = torch.stack(
                [torch.randperm(n_patches, generator=g) for _ in ib]).to(device)

        for name, order in orders.items():
            for frac in args.fractions:
                n = max(1, int(round(frac * n_patches)))
                mk = patch_mask(order, n)
                lg, pr = score(px * (1 - mk) + fill * mk)
                for j, i in enumerate(ib):
                    rows.append({
                        "image_name": names[i], "correct": bool(correct[i]),
                        "method": "random" if name.startswith("random") else name,
                        "seed": name, "fraction": frac,
                        "logit_drop": float(base_logit[j] - lg[j]),
                        "prob_drop": float(base_prob[j] - pr[j]),
                    })

    long = pd.DataFrame(rows)
    # average random over seeds first, per image
    per_img = (long.groupby(["image_name", "correct", "method", "fraction"],
                            as_index=False)[["logit_drop", "prob_drop"]].mean())
    per_img.to_csv(out_dir / "deletion_long.csv", index=False)

    rnd = per_img[per_img.method == "random"].set_index(["image_name", "fraction"])
    summ = []
    for m in methods:
        d = per_img[per_img.method == m].set_index(["image_name", "fraction"])
        diff = d["logit_drop"] - rnd.loc[d.index, "logit_drop"]
        by_k = diff.groupby(level="fraction")
        summ.append({
            "method": m,
            "mean_logit_drop": d["logit_drop"].mean(),
            "random_mean_logit_drop": rnd.loc[d.index, "logit_drop"].mean(),
            "mean_gain_over_random": diff.mean(),       # main selection score
            "frac_images_beating_random_at_10pct":
                float((diff.xs(0.10, level="fraction") > 0).mean())
                if 0.10 in args.fractions else np.nan,
            **{f"gain_at_{k:g}": v for k, v in by_k.mean().items()},
        })
    summ = pd.DataFrame(summ).sort_values("mean_gain_over_random", ascending=False)
    summ.to_csv(out_dir / "deletion_summary.csv", index=False)
    print("\nDeletion summary (higher gain over random = more faithful):")
    print(summ.round(3).to_string(index=False))

    # ---------------- plot ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = per_img.groupby(["method", "fraction"])["logit_drop"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in curves.columns:
        style = dict(color="black", ls="--", lw=2) if m == "random" else dict(marker="o")
        ax.plot(curves.index * 100, curves[m], label=m, **style)
    ax.set_xlabel("% of patches removed (highest attribution first)")
    ax.set_ylabel(f"Drop in {args.which}-class logit")
    ax.set_title(f"Deletion test, CLIP ViT-B/16 on ExDark ({len(sel)} images, fill={args.fill})")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "deletion_curves.png", dpi=200)
    plt.close()
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
