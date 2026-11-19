"""
STEP 4b - Object Attribution Lift for every image, every method (CPU only).

OAL = (fraction of attribution inside the target mask) / (target area fraction)
both measured in CLIP's 224x224 input frame, with the 14x14 map bilinearly
upsampled to 224x224 and the soft (area-averaged) box mask.

Also computes, for the predicted-class map:
  attrin_allobj_*   share of evidence on ANY annotated object
  oal_predobj_*     OAL w.r.t. boxes of the PREDICTED class, when that class is
                    annotated in the image (e.g. Cup->Table with a labelled
                    table): does the map sit on the object it named?

  python compute_oal.py --tag pilot
  python compute_oal.py --tag full

Output: outputs/exdark/attribution/<tag>/oal_table.csv  (one row per image)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exdark_common as C  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--chunk", type=int, default=256)
    args = p.parse_args(argv)

    run_dir = C.ATTR_DIR / args.tag
    z = np.load(run_dir / "maps.npz")
    ids = z["image_id"].astype(str)
    names = z["image_name"].astype(str)
    methods = sorted({k.split("__")[0] for k in z.files if "__" in k})

    mz = np.load(C.CROPPED_MASK_DIR / "testing_masks.npz")
    pos = {k: i for i, k in enumerate(mz["image_id"].astype(str))}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise RuntimeError(f"{len(missing)} images have no cropped mask; "
                           "run make_cropped_masks.py first")
    sel = np.array([pos[i] for i in ids])
    tgt = mz["target"][sel].astype(np.float32) / 255.0
    allm = mz["all_objects"][sel].astype(np.float32) / 255.0

    stats = pd.read_csv(C.CROPPED_MASK_DIR / "testing_mask_stats.csv")
    meta = pd.read_csv(C.IMAGE_CSV).set_index("image_name")
    objs = pd.read_csv(C.OBJECT_CSV)
    objs["object_class_l"] = objs["object_class"].str.lower()
    obj_groups = dict(tuple(objs.groupby("image_name")))
    crop_mode = stats["crop_mode"].iloc[0]

    pred_cls = [C.CLASS_NAMES[i] for i in z["pred_idx"]]
    tgt_cls = [C.CLASS_NAMES[i] for i in z["target_idx"]]

    out = pd.DataFrame({"image_name": names, "image_id": ids,
                        "ground_truth": tgt_cls, "prediction": pred_cls})
    out["correct"] = out["ground_truth"] == out["prediction"]
    lg = z["logits"]
    out["pred_logit"] = lg[np.arange(len(lg)), z["pred_idx"]]
    out["target_logit"] = lg[np.arange(len(lg)), z["target_idx"]]
    out["logit_margin"] = out["target_logit"] - out["pred_logit"]

    # ---- predicted class present among the annotated objects? ----------
    pred_annot, predobj_masks = [], {}
    for i, (n, pc) in enumerate(zip(names, pred_cls)):
        o = obj_groups.get(n)
        hit = o is not None and (o["object_class_l"] == pc.lower()).any()
        pred_annot.append(bool(hit))
        if hit and pc != tgt_cls[i]:
            b = o[o["object_class_l"] == pc.lower()][["x1", "y1", "x2", "y2"]].to_numpy()
            full = C.boxes_to_mask(b, int(meta.loc[n, "image_width"]),
                                   int(meta.loc[n, "image_height"]))
            predobj_masks[i], _ = C.mask_to_clip_frame(full, crop_mode)
    out["pred_class_annotated"] = pred_annot
    out["error_type"] = np.where(
        out["correct"], "correct",
        np.where(out["pred_class_annotated"], "named_other_annotated_object",
                 "named_unannotated_class"))

    # ---- OAL for every method / class ---------------------------------
    for m in tqdm(methods, desc="OAL"):
        for which in ("pred", "target"):
            maps = z[f"{m}__{which}"]
            oal, inside, allin, predobj = [], [], [], []
            for s in range(0, len(maps), args.chunk):
                up = C.upsample(maps[s: s + args.chunk])
                for j, h in enumerate(up):
                    i = s + j
                    a_in, _, o = C.oal_stats(h, tgt[i])
                    oal.append(o)
                    inside.append(a_in)
                    if which == "pred":
                        allin.append(C.oal_stats(h, allm[i])[0])
                        predobj.append(C.oal_stats(h, predobj_masks[i])[2]
                                       if i in predobj_masks else np.nan)
            out[f"oal_{m}_{which}"] = oal
            out[f"attrin_{m}_{which}"] = inside
            if which == "pred":
                out[f"attrin_allobj_{m}"] = allin
                out[f"oal_predobj_{m}"] = predobj

    keep_meta = meta[["lighting", "environment", "n_target_objects",
                      "n_valid_objects", "image_width", "image_height"]]
    out = (out.merge(keep_meta, left_on="image_name", right_index=True, how="left")
              .merge(stats.drop(columns=["image_id"]), on="image_name", how="left"))

    out.to_csv(run_dir / "oal_table.csv", index=False)
    print(f"Wrote {run_dir / 'oal_table.csv'}  ({len(out)} rows, {len(methods)} methods)")

    usable = out[~out["no_target_box"] & ~out["target_cropped_out"]]
    cols = [f"oal_{m}_pred" for m in methods]
    print("\nMedian OAL (predicted-class maps), correct vs incorrect:")
    print(usable.groupby("correct")[cols].median().T.round(3).to_string())


if __name__ == "__main__":
    main()
