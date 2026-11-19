"""
STEP 1a - Pick a fixed, stratified pilot set (~60 test images).

Per class: up to 3 correct + 2 incorrect predictions, spread over target size
(one small, one medium, one large where possible). Also forces in the
highest-confidence Cup->Table, Bottle->Table, Chair->People, Dog->Cat errors
used in the progress slides. Skips images with no target box or a target
that is mostly cropped out.

Output: outputs/exdark/attribution/pilot_manifest.csv

  python make_pilot_manifest.py --per-class-correct 3 --per-class-incorrect 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exdark_common as C  # noqa: E402

FORCED_ERRORS = [("Cup", "Table"), ("Bottle", "Table"),
                 ("Chair", "People"), ("Dog", "Cat")]


def spread_by_size(df, k, rng):
    """Pick k rows spread across the target-size distribution."""
    if len(df) <= k:
        return df
    df = df.sort_values("crop_target_area_fraction").reset_index(drop=True)
    bins = np.array_split(np.arange(len(df)), k)
    return df.iloc[[rng.choice(b) for b in bins]]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--per-class-correct", type=int, default=3)
    p.add_argument("--per-class-incorrect", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(C.ATTR_DIR / "pilot_manifest.csv"))
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    pred = pd.read_csv(C.PREDICTIONS_CSV)
    stats = pd.read_csv(C.CROPPED_MASK_DIR / "testing_mask_stats.csv")
    df = pred.merge(stats[["image_name", "crop_target_area_fraction",
                           "target_crop_survival", "no_target_box", "low_survival"]],
                    on="image_name", how="left")
    df["correct"] = df["ground_truth"] == df["prediction"]
    usable = df[~df["no_target_box"] & ~df["low_survival"]]

    picks = []
    for gt, pr in FORCED_ERRORS:
        c = usable[(usable.ground_truth == gt) & (usable.prediction == pr)]
        if len(c):
            picks.append(c.sort_values("confidence", ascending=False).head(1)
                         .assign(reason=f"forced {gt}->{pr}"))

    for cls in C.CLASS_NAMES:
        sub = usable[usable.ground_truth == cls]
        picks.append(spread_by_size(sub[sub.correct], args.per_class_correct, rng)
                     .assign(reason="correct"))
        picks.append(spread_by_size(sub[~sub.correct], args.per_class_incorrect, rng)
                     .assign(reason="incorrect"))

    man = (pd.concat(picks).drop_duplicates("image_name")
           .sort_values(["ground_truth", "correct"]).reset_index(drop=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    man[["image_name", "image_id", "ground_truth", "prediction", "correct",
         "confidence", "lighting", "crop_target_area_fraction",
         "target_crop_survival", "reason"]].to_csv(args.out, index=False)

    print(f"Pilot images: {len(man)}  (correct={int(man.correct.sum())}, "
          f"incorrect={int((~man.correct).sum())})")
    print(man.groupby("ground_truth").correct.agg(["size", "sum"]).to_string())
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
