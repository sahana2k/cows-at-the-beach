from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path.home() / "cse598_vlm_project"

IMAGE_CSV = (
    PROJECT_ROOT
    / "data/exdark/processed/metadata/exdark_images.csv"
)

OBJECT_CSV = (
    PROJECT_ROOT
    / "data/exdark/processed/metadata/exdark_objects.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/exdark/presentation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


images_df = pd.read_csv(IMAGE_CSV)
objects_df = pd.read_csv(OBJECT_CSV)


# ============================================================
# VISUAL 1
# ExDark class distribution
# ============================================================

class_counts = (
    images_df["class_name"]
    .value_counts()
    .sort_values()
)

plt.figure(figsize=(9, 6))

class_counts.plot(
    kind="barh"
)

plt.xlabel("Number of images")
plt.ylabel("Class")

plt.title(
    "ExDark class distribution\n"
    "7,363 naturally low-light images"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_exdark_class_distribution.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# VISUAL 2
# Lighting-condition distribution
# ============================================================

lighting_counts = (
    images_df["lighting"]
    .value_counts()
    .sort_values()
)

plt.figure(figsize=(9, 5.5))

lighting_counts.plot(
    kind="barh"
)

plt.xlabel("Number of images")
plt.ylabel("Lighting condition")

plt.title(
    "ExDark lighting-condition distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_exdark_lighting_distribution.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# VISUAL 3
# Train / validation / test split
# ============================================================

split_order = [
    "Training",
    "Validation",
    "Testing",
]

split_counts = (
    images_df["split"]
    .value_counts()
    .reindex(split_order)
)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    split_counts.index,
    split_counts.values,
)

plt.ylabel("Number of images")

plt.title(
    "Official ExDark experimental split"
)

for bar, value in zip(
    bars,
    split_counts.values,
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 40,
        f"{value:,}",
        ha="center",
        fontsize=11,
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_exdark_split.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# VISUAL 4
# Three-example preprocessing figure
# ============================================================

EXAMPLE_IDS = [
    "2015_03355",   # Cat
    "2015_00640",   # Bicycle
    "2015_05971",   # Motorbike
]


example_rows = (
    images_df[
        images_df["image_id"].isin(EXAMPLE_IDS)
    ]
    .set_index("image_id")
    .loc[EXAMPLE_IDS]
    .reset_index()
)


fig, axes = plt.subplots(
    len(example_rows),
    3,
    figsize=(12, 10),
)


for row_idx, row in example_rows.iterrows():

    image = Image.open(
        row["image_path"]
    ).convert("RGB")

    boxed = image.copy()

    draw = ImageDraw.Draw(boxed)

    image_objects = objects_df[
        objects_df["image_name"]
        == row["image_name"]
    ]

    # Only emphasize target-class boxes
    target_objects = image_objects[
        image_objects["is_target_class"] == True
    ]

    for _, obj in target_objects.iterrows():

        x1 = int(obj["x1"])
        y1 = int(obj["y1"])
        x2 = int(obj["x2"])
        y2 = int(obj["y2"])

        for offset in range(3):

            draw.rectangle(
                [
                    x1 - offset,
                    y1 - offset,
                    x2 + offset,
                    y2 + offset,
                ],
                outline="red",
            )

    mask = Image.open(
        row["target_mask_path"]
    ).convert("L")

    axes[row_idx, 0].imshow(image)

    axes[row_idx, 0].set_ylabel(
        row["class_name"],
        fontsize=12,
        fontweight="bold",
    )

    axes[row_idx, 1].imshow(boxed)

    axes[row_idx, 2].imshow(
        mask,
        cmap="gray",
    )

    area = (
        row["target_area_fraction"]
        * 100
    )

    axes[row_idx, 2].set_title(
        f"Target area = {area:.1f}%"
    )

    for col in range(3):
        axes[row_idx, col].set_xticks([])
        axes[row_idx, col].set_yticks([])


axes[0, 0].set_title(
    "Original ExDark image",
    fontsize=12,
)

axes[0, 1].set_title(
    "Bounding-box annotation",
    fontsize=12,
)

# First mask title needs both column title and area
first_area = (
    example_rows.iloc[0]["target_area_fraction"]
    * 100
)

axes[0, 2].set_title(
    f"Target-region mask\n"
    f"Target area = {first_area:.1f}%",
    fontsize=12,
)


fig.suptitle(
    "ExDark preprocessing for object-attribution analysis",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_exdark_preprocessing_examples.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


print()
print("Presentation figures created:")

for path in sorted(
    OUTPUT_DIR.glob("*.png")
):
    print(path)
