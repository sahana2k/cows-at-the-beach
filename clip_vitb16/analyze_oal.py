"""
STEP 6 - RQ1 analysis: does target-object attribution differ between correct
and incorrect predictions on ExDark?

Run AFTER you have chosen a method with the deletion test:

  python analyze_oal.py --tag full --method gradcam_L11

Outputs in outputs/exdark/attribution/<tag>/analysis_<method>/:
  rq1_tests.csv          Mann-Whitney U, Cliff's delta, bootstrap CI, OAL>1 tests
  size_stratified.csv    same test within target-area quartiles (size confound)
  logistic_*.txt         correct ~ log(OAL) + log(area) + class (+ lighting)
  per_class.csv          accuracy and median OAL per class
  lighting.csv           descriptive only (different images per category!)
  error_types.csv        do errors name another annotated object in the scene?
  fig_*.png              violin, OAL vs area, per-class, error breakdown
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exdark_common as ec  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EPS = 0.01  # added before log(OAL) because OAL can be exactly 0


def compare(x, y, n_boot=2000, seed=0):
    """x = correct, y = incorrect."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    res = {"n_correct": len(x), "n_incorrect": len(y)}
    if len(x) < 3 or len(y) < 3:
        return res
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    rng = np.random.default_rng(seed)
    boots = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y)))
             for _ in range(n_boot)]
    res.update({
        "median_correct": np.median(x), "median_incorrect": np.median(y),
        "median_diff": np.median(x) - np.median(y),
        "median_diff_ci_lo": np.percentile(boots, 2.5),
        "median_diff_ci_hi": np.percentile(boots, 97.5),
        "mannwhitney_U": U, "p_value": p,
        "cliffs_delta": 2 * U / (len(x) * len(y)) - 1,   # >0: correct higher
        "frac_oal_gt1_correct": float((x > 1).mean()),
        "frac_oal_gt1_incorrect": float((y > 1).mean()),
        "wilcoxon_vs1_p_correct": stats.wilcoxon(np.log(x + EPS) - np.log(1 + EPS)).pvalue,
        "wilcoxon_vs1_p_incorrect": stats.wilcoxon(np.log(y + EPS) - np.log(1 + EPS)).pvalue,
    })
    return res


def logistic(df, col, out_path):
    import statsmodels.formula.api as smf
    d = df[[col, "correct", "crop_target_area_fraction", "ground_truth", "lighting"]].dropna().copy()
    d["log_oal"] = np.log(d[col] + EPS)
    d["log_area"] = np.log(d["crop_target_area_fraction"])
    d["y"] = d["correct"].astype(int)
    lines = []
    for f in ["y ~ log_oal",
              "y ~ log_oal + log_area",
              "y ~ log_oal + log_area + C(ground_truth)",
              "y ~ log_oal + log_area + C(ground_truth) + C(lighting)"]:
        try:
            r = smf.logit(f, d).fit(disp=0, maxiter=200)
            ci = r.conf_int().loc["log_oal"]
            lines.append(f"{f}\n  log_oal coef={r.params['log_oal']:.3f} "
                         f"[{ci[0]:.3f}, {ci[1]:.3f}]  p={r.pvalues['log_oal']:.2e}  "
                         f"OR per e-fold OAL={np.exp(r.params['log_oal']):.3f}  "
                         f"pseudo-R2={r.prsquared:.3f}  n={int(r.nobs)}\n")
            if f.endswith("C(lighting)"):
                lines.append(str(r.summary()) + "\n")
        except Exception as e:  # e.g. perfect separation in a small pilot
            lines.append(f"{f}\n  failed: {e}\n")
    Path(out_path).write_text("\n".join(lines))
    print("".join(lines[:4]))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--method", required=True, help="e.g. gradcam_L11 or chefer_full")
    p.add_argument("--min-survival", type=float, default=0.5)
    args = p.parse_args(argv)

    run_dir = ec.ATTR_DIR / args.tag
    out = run_dir / f"analysis_{args.method}"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_dir / "oal_table.csv")
    m = args.method
    usable = df[~df["no_target_box"] & ~df["target_cropped_out"]].copy()
    strict = usable[usable["target_crop_survival"] >= args.min_survival]
    print(f"Images: {len(df)} | usable (has target in crop): {len(usable)} | "
          f"survival>={args.min_survival}: {len(strict)}")

    # ---------------- main test ----------------
    rows = []
    for which in ("pred", "target"):
        col = f"oal_{m}_{which}"
        for name, d in (("usable", usable), (f"survival>={args.min_survival}", strict)):
            r = compare(d.loc[d.correct, col], d.loc[~d.correct, col])
            rows.append({"map_for": which, "subset": name, **r})
    rq1 = pd.DataFrame(rows)
    rq1.to_csv(out / "rq1_tests.csv", index=False)
    print("\nRQ1: OAL correct vs incorrect")
    print(rq1[["map_for", "subset", "n_correct", "n_incorrect", "median_correct",
               "median_incorrect", "cliffs_delta", "p_value"]].round(4).to_string(index=False))

    # ---------------- size confound ----------------
    usable["area_quartile"] = pd.qcut(usable["crop_target_area_fraction"], 4,
                                      labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
    rows = []
    for q, d in usable.groupby("area_quartile", observed=True):
        for which in ("pred", "target"):
            col = f"oal_{m}_{which}"
            rows.append({"area_quartile": q, "map_for": which,
                         "accuracy": d.correct.mean(),
                         "area_range": f"{d.crop_target_area_fraction.min():.3f}-"
                                       f"{d.crop_target_area_fraction.max():.3f}",
                         **compare(d.loc[d.correct, col], d.loc[~d.correct, col], n_boot=500)})
    pd.DataFrame(rows).to_csv(out / "size_stratified.csv", index=False)
    rho = stats.spearmanr(usable["crop_target_area_fraction"], usable[f"oal_{m}_pred"],
                          nan_policy="omit")
    print(f"\nSpearman(area, OAL_pred) = {rho.correlation:.3f}  (expect negative: "
          "small boxes allow large OAL)")

    for which in ("pred", "target"):
        print(f"\nLogistic regression, map for {which}:")
        logistic(usable, f"oal_{m}_{which}", out / f"logistic_{which}.txt")

    # ---------------- per class / lighting ----------------
    agg = dict(n=("correct", "size"), accuracy=("correct", "mean"),
               median_oal_pred=(f"oal_{m}_pred", "median"),
               median_oal_target=(f"oal_{m}_target", "median"),
               median_area=("crop_target_area_fraction", "median"))
    per_class = usable.groupby("ground_truth").agg(**agg).sort_values("accuracy")
    per_class.to_csv(out / "per_class.csv")
    usable.groupby("lighting").agg(**agg).sort_values("accuracy").to_csv(out / "lighting.csv")
    print("\nPer class:\n", per_class.round(3).to_string())

    # ---------------- error types ----------------
    err = usable[~usable.correct]
    et = (err.groupby("ground_truth")["error_type"].value_counts().unstack(fill_value=0))
    et["n_errors"] = et.sum(1)
    if "named_other_annotated_object" in et:
        et["frac_named_other_annotated"] = et["named_other_annotated_object"] / et["n_errors"]
    et.loc["ALL"] = et.sum(numeric_only=True)
    if "named_other_annotated_object" in et:
        et.loc["ALL", "frac_named_other_annotated"] = (
            et.loc["ALL", "named_other_annotated_object"] / et.loc["ALL", "n_errors"])
    et.to_csv(out / "error_types.csv")
    print("\nError types:\n", et.round(3).to_string())

    other = err[err.error_type == "named_other_annotated_object"]
    # predicted-object box can be cropped out of CLIP's view -> NaN; drop those pairs
    other = other.dropna(subset=[f"oal_predobj_{m}", f"oal_{m}_pred"])
    if len(other) >= 3:
        w = stats.wilcoxon(np.log(other[f"oal_predobj_{m}"] + EPS),
                           np.log(other[f"oal_{m}_pred"] + EPS))
        frac_more = float((other[f"oal_predobj_{m}"] > other[f"oal_{m}_pred"]).mean())
        msg = (f"Errors naming another annotated object (n={len(other)} with both boxes "
               f"in view): median OAL of pred-class map on PREDICTED object boxes = "
               f"{other[f'oal_predobj_{m}'].median():.2f} vs on TARGET boxes = "
               f"{other[f'oal_{m}_pred'].median():.2f} (Wilcoxon p={w.pvalue:.2e}); "
               f"map favours the predicted object in {frac_more:.0%} of these errors")
        print("\n" + msg)
        (out / "predobj_vs_target.txt").write_text(msg + "\n")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, which in zip(axes, ("pred", "target")):
        col = f"oal_{m}_{which}"
        data = [np.log10(usable.loc[usable.correct, col].dropna() + EPS),
                np.log10(usable.loc[~usable.correct, col].dropna() + EPS)]
        ax.violinplot(data, showmedians=True)
        ax.axhline(0, color="grey", ls="--", lw=1)
        r = rq1[(rq1.map_for == which) & (rq1.subset == "usable")].iloc[0]
        ax.set_xticks([1, 2], [f"Correct\n(n={len(data[0])})", f"Incorrect\n(n={len(data[1])})"])
        ax.set_title(f"map for {which} class\nCliff's δ={r.get('cliffs_delta', np.nan):.2f}, "
                     f"p={r.get('p_value', np.nan):.1e}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("log10 OAL   (0 = attribution proportional to area)")
    fig.suptitle(f"CLIP ViT-B/16 on ExDark: OAL by correctness ({m})")
    plt.tight_layout()
    plt.savefig(out / "fig_rq1_violin.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ok, color in ((True, "tab:blue"), (False, "tab:red")):
        d = usable[usable.correct == ok]
        ax.scatter(d.crop_target_area_fraction, d[f"oal_{m}_pred"] + EPS, s=6,
                   alpha=0.35, color=color, label="correct" if ok else "incorrect")
    xs = np.logspace(np.log10(usable.crop_target_area_fraction.min()), 0, 50)
    ax.plot(xs, 1 / xs, "k:", lw=1, label="max possible OAL (=1/area)")
    ax.set(xscale="log", yscale="log", xlabel="Target area fraction (224x224 crop)",
           ylabel="OAL (pred-class map)", title="OAL vs target size")
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_oal_vs_area.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pc = per_class.sort_values("accuracy")
    ax.scatter(pc.accuracy * 100, pc.median_oal_pred, s=40)
    for c, r in pc.iterrows():
        ax.annotate(c, (r.accuracy * 100, r.median_oal_pred), fontsize=8,
                    xytext=(4, 2), textcoords="offset points")
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set(xlabel="Class accuracy (%)", ylabel="Median OAL (pred-class map)",
           title="Per-class accuracy vs object focus")
    plt.tight_layout()
    plt.savefig(out / "fig_per_class.png", dpi=200)
    plt.close()

    if "frac_named_other_annotated" in et:
        e = et.drop("ALL").sort_values("n_errors")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        a = e.get("named_other_annotated_object", 0)
        b = e.get("named_unannotated_class", 0)
        ax.barh(e.index, a, label="named another annotated object")
        ax.barh(e.index, b, left=a, label="named a class not annotated")
        ax.set(xlabel="Number of errors", title="What do CLIP's ExDark errors name?")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "fig_error_types.png", dpi=200)
        plt.close()

    print("\nWrote", out)


if __name__ == "__main__":
    main()