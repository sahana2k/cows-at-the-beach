from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path.home() / "cse598_vlm_project"

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs/exdark/clip_vitb16_testing_predictions.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data/exdark/processed/metadata/exdark_images.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/exdark/presentation/results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

CLASS_ORDER = [
    "Bicycle",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Motorbike",
    "People",
    "Table",
]


# ============================================================
# LOAD DATA
# ============================================================

pred = pd.read_csv(PREDICTIONS_PATH)

metadata = pd.read_csv(METADATA_PATH)


print("=" * 70)
print("CLIP ViT-B/16 ExDark Analysis")
print("=" * 70)

print("Predictions:", len(pred))


# ============================================================
# VALIDATE RESULTS
# ============================================================

assert len(pred) == 2563, (
    f"Expected 2563 test predictions, "
    f"found {len(pred)}"
)

assert pred["image_name"].nunique() == 2563, (
    "Duplicate image predictions detected."
)

assert set(pred["ground_truth"]).issubset(
    set(CLASS_ORDER)
)

assert set(pred["prediction"]).issubset(
    set(CLASS_ORDER)
)


# Recompute correctness ourselves rather than blindly
# trusting the existing column.

pred["correct_recomputed"] = (
    pred["ground_truth"]
    == pred["prediction"]
)


if "correct" in pred.columns:

    existing_correct = (
        pred["correct"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    agreement = (
        existing_correct
        == pred["correct_recomputed"]
    ).all()

    print(
        "Saved/recomputed correctness agree:",
        agreement
    )


# ============================================================
# OVERALL ACCURACY
# ============================================================

n_total = len(pred)

n_correct = int(
    pred["correct_recomputed"].sum()
)

overall_accuracy = (
    n_correct / n_total
)

print()
print("Overall results")
print("------------------------------")

print("N:", n_total)
print("Correct:", n_correct)

print(
    "Accuracy:",
    f"{overall_accuracy:.4f}",
    f"({overall_accuracy * 100:.2f}%)",
)


# ============================================================
# CLASS ACCURACY
# ============================================================

class_results = (
    pred
    .groupby("ground_truth")
    .agg(
        n=("image_name", "size"),
        correct=(
            "correct_recomputed",
            "sum",
        ),
    )
)

class_results["accuracy"] = (
    class_results["correct"]
    / class_results["n"]
)

class_results["accuracy_percent"] = (
    class_results["accuracy"] * 100
)

class_results = (
    class_results
    .reindex(CLASS_ORDER)
)

class_results.to_csv(
    OUTPUT_DIR / "class_accuracy.csv"
)


# ============================================================
# LIGHTING ACCURACY
# ============================================================

lighting_results = (
    pred
    .groupby("lighting")
    .agg(
        n=("image_name", "size"),
        correct=(
            "correct_recomputed",
            "sum",
        ),
    )
)

lighting_results["accuracy"] = (
    lighting_results["correct"]
    / lighting_results["n"]
)

lighting_results["accuracy_percent"] = (
    lighting_results["accuracy"]
    * 100
)

lighting_results = (
    lighting_results
    .sort_values(
        "accuracy_percent",
        ascending=False,
    )
)

lighting_results.to_csv(
    OUTPUT_DIR / "lighting_accuracy.csv"
)


# ============================================================
# ENVIRONMENT ACCURACY
# ============================================================

environment_results = (
    pred
    .groupby("environment")
    .agg(
        n=("image_name", "size"),
        correct=(
            "correct_recomputed",
            "sum",
        ),
    )
)

environment_results["accuracy"] = (
    environment_results["correct"]
    / environment_results["n"]
)

environment_results[
    "accuracy_percent"
] = (
    environment_results["accuracy"]
    * 100
)

environment_results.to_csv(
    OUTPUT_DIR / "environment_accuracy.csv"
)


# ============================================================
# PRINT TABLES
# ============================================================

print()
print("Accuracy by class")
print("------------------------------")

print(
    class_results[
        ["n", "correct", "accuracy_percent"]
    ]
    .round(2)
    .to_string()
)


print()
print("Accuracy by lighting")
print("------------------------------")

print(
    lighting_results[
        ["n", "correct", "accuracy_percent"]
    ]
    .round(2)
    .to_string()
)


# ============================================================
# FIGURE 1:
# ACCURACY BY CLASS
# ============================================================

plot_class = (
    class_results
    .reset_index()
    .rename(
        columns={
            "ground_truth": "class_name"
        }
    )
    .sort_values(
        "accuracy_percent",
        ascending=True,
    )
)


fig, ax = plt.subplots(
    figsize=(9, 6.5)
)

bars = ax.barh(
    plot_class["class_name"],
    plot_class["accuracy_percent"],
)


ax.axvline(
    overall_accuracy * 100,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Overall = "
        f"{overall_accuracy * 100:.1f}%"
    ),
)


ax.set_xlim(0, 105)

ax.set_xlabel(
    "Zero-shot accuracy (%)"
)

ax.set_ylabel("")

ax.set_title(
    "CLIP ViT-B/16 accuracy varies substantially by object class",
    fontsize=14,
    fontweight="bold",
)


for bar, value in zip(
    bars,
    plot_class["accuracy_percent"],
):

    ax.text(
        value + 1,
        bar.get_y()
        + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        fontsize=9,
    )


ax.legend(
    loc="lower right"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "05_clip_accuracy_by_class.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 2:
# ACCURACY BY LIGHTING CONDITION
# ============================================================

lighting_plot = (
    lighting_results
    .reset_index()
    .sort_values(
        "accuracy_percent",
        ascending=True,
    )
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.barh(
    lighting_plot["lighting"],
    lighting_plot["accuracy_percent"],
)


ax.axvline(
    overall_accuracy * 100,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Overall = "
        f"{overall_accuracy * 100:.1f}%"
    ),
)


ax.set_xlim(0, 105)

ax.set_xlabel(
    "Zero-shot accuracy (%)"
)

ax.set_ylabel("")

ax.set_title(
    "CLIP ViT-B/16 accuracy across ExDark lighting categories",
    fontsize=14,
    fontweight="bold",
)


for bar, accuracy, n in zip(
    bars,
    lighting_plot["accuracy_percent"],
    lighting_plot["n"],
):

    ax.text(
        accuracy + 1,
        bar.get_y()
        + bar.get_height() / 2,
        f"{accuracy:.1f}%  (n={n})",
        va="center",
        fontsize=8.5,
    )


ax.legend(
    loc="lower right"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "06_clip_accuracy_by_lighting.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    pred["ground_truth"],
    pred["prediction"],
    labels=CLASS_ORDER,
)


# Raw confusion matrix

cm_df = pd.DataFrame(
    cm,
    index=CLASS_ORDER,
    columns=CLASS_ORDER,
)

cm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix_counts.csv"
)


# Normalize each ground-truth row

row_sums = cm.sum(
    axis=1,
    keepdims=True,
)

cm_norm = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(
        cm,
        dtype=float,
    ),
    where=row_sums != 0,
)


fig, ax = plt.subplots(
    figsize=(10, 9)
)

image = ax.imshow(
    cm_norm,
    vmin=0,
    vmax=1,
    cmap="Blues",
)


ax.set_xticks(
    np.arange(len(CLASS_ORDER))
)

ax.set_yticks(
    np.arange(len(CLASS_ORDER))
)

ax.set_xticklabels(
    CLASS_ORDER,
    rotation=45,
    ha="right",
)

ax.set_yticklabels(
    CLASS_ORDER
)


ax.set_xlabel(
    "Predicted class"
)

ax.set_ylabel(
    "Ground-truth class"
)

ax.set_title(
    "CLIP ViT-B/16 normalized confusion matrix",
    fontsize=14,
    fontweight="bold",
)


for i in range(len(CLASS_ORDER)):

    for j in range(len(CLASS_ORDER)):

        value = cm_norm[i, j]

        if value >= 0.10:

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )


fig.colorbar(
    image,
    ax=ax,
    label="Fraction of ground-truth class",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "07_clip_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# MOST COMMON ERRORS
# ============================================================

errors = pred[
    ~pred["correct_recomputed"]
].copy()


common_errors = (
    errors
    .groupby(
        [
            "ground_truth",
            "prediction",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False,
    )
)


common_errors.to_csv(
    OUTPUT_DIR
    / "common_errors.csv",
    index=False,
)


print()
print("Most common errors")
print("------------------------------")

print(
    common_errors
    .head(15)
    .to_string(index=False)
)


# ============================================================
# ADD IMAGE PATHS TO PREDICTIONS
# ============================================================

image_lookup = (
    metadata[
        [
            "image_name",
            "image_path",
        ]
    ]
    .drop_duplicates(
        subset="image_name"
    )
)


pred_with_paths = pred.merge(
    image_lookup,
    on="image_name",
    how="left",
)


assert (
    pred_with_paths[
        "image_path"
    ].notna().all()
), "Some prediction images could not be matched."


# ============================================================
# QUALITATIVE ERROR EXAMPLES
#
# Prefer common confusion types so the examples tell
# a coherent story rather than being random failures.
# ============================================================

preferred_errors = [
    ("Cup", "Table"),
    ("Bottle", "Table"),
    ("Chair", "People"),
    ("Dog", "Cat"),
]


selected_errors = []


for ground_truth, prediction in preferred_errors:

    candidates = pred_with_paths[
        (
            pred_with_paths[
                "ground_truth"
            ] == ground_truth
        )
        &
        (
            pred_with_paths[
                "prediction"
            ] == prediction
        )
    ]

    if len(candidates) > 0:

        # Choose highest normalized class score
        # among this error type so the example
        # is a clear model decision.

        chosen = candidates.sort_values(
            "confidence",
            ascending=False,
        ).iloc[0]

        selected_errors.append(
            chosen
        )


# If fewer than 4 preferred examples exist,
# fill with high-confidence errors.

if len(selected_errors) < 4:

    existing_names = {
        row["image_name"]
        for row in selected_errors
    }

    remaining = (
        pred_with_paths[
            ~pred_with_paths[
                "correct_recomputed"
            ]
        ]
        .sort_values(
            "confidence",
            ascending=False,
        )
    )

    for _, row in remaining.iterrows():

        if (
            row["image_name"]
            not in existing_names
        ):

            selected_errors.append(
                row
            )

            existing_names.add(
                row["image_name"]
            )

        if len(selected_errors) == 4:
            break


# ============================================================
# FIGURE 4:
# QUALITATIVE ERROR EXAMPLES
# ============================================================

fig, axes = plt.subplots(
    1,
    len(selected_errors),
    figsize=(14, 4.3),
)


if len(selected_errors) == 1:
    axes = [axes]


for ax, row in zip(
    axes,
    selected_errors,
):

    image = Image.open(
        row["image_path"]
    ).convert("RGB")

    ax.imshow(image)

    ax.set_title(
        f"ExDark target: {row['ground_truth']}\n"
        f"CLIP: {row['prediction']}",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        0.5,
        -0.08,
        (
            f"{row['lighting']} lighting"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.axis("off")


fig.suptitle(
    "Examples where CLIP disagrees with the ExDark target label",
    fontsize=15,
    fontweight="bold",
)


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "08_clip_error_examples.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# CORRECT VS INCORRECT EXAMPLES
# ============================================================

correct_examples = (
    pred_with_paths[
        pred_with_paths[
            "correct_recomputed"
        ]
    ]
    .sort_values(
        "confidence",
        ascending=False,
    )
    .head(3)
)


incorrect_examples = (
    pred_with_paths[
        ~pred_with_paths[
            "correct_recomputed"
        ]
    ]
    .sort_values(
        "confidence",
        ascending=False,
    )
    .head(3)
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(11, 7),
)


for col, (_, row) in enumerate(
    correct_examples.iterrows()
):

    image = Image.open(
        row["image_path"]
    ).convert("RGB")

    axes[0, col].imshow(image)

    axes[0, col].set_title(
        f"ExDark target: {row['ground_truth']}\n"
        f"CLIP: {row['prediction']} ✓",
        fontsize=10,
    )

    axes[0, col].axis("off")


for col, (_, row) in enumerate(
    incorrect_examples.iterrows()
):

    image = Image.open(
        row["image_path"]
    ).convert("RGB")

    axes[1, col].imshow(image)

    axes[1, col].set_title(
        f"ExDark target: {row['ground_truth']}\n"
        f"CLIP: {row['prediction']} ✗",
        fontsize=10,
    )

    axes[1, col].axis("off")


axes[0, 0].set_ylabel(
    "Correct",
    fontsize=12,
    fontweight="bold",
)

axes[1, 0].set_ylabel(
    "Incorrect",
    fontsize=12,
    fontweight="bold",
)


fig.suptitle(
    "Correct and incorrect CLIP predictions on ExDark",
    fontsize=15,
    fontweight="bold",
)


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "09_correct_vs_incorrect_examples.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "model": "CLIP ViT-B/16",
            "dataset": "ExDark",
            "split": "Testing",
            "n": n_total,
            "correct": n_correct,
            "accuracy": overall_accuracy,
            "accuracy_percent":
                overall_accuracy * 100,
        }
    ]
)


summary.to_csv(
    OUTPUT_DIR
    / "clip_results_summary.csv",
    index=False,
)


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Overall accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print()
print("Files written:")

for path in sorted(
    OUTPUT_DIR.iterdir()
):
    print(" ", path.name)
