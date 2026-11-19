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

QA_DIR = (
    PROJECT_ROOT
    / "data/exdark/processed/qa"
)

QA_DIR.mkdir(parents=True, exist_ok=True)


def main():

    images = pd.read_csv(IMAGE_CSV)
    objects = pd.read_csv(OBJECT_CSV)

    # --------------------------------------------------------
    # Select one reproducible example from every class
    # --------------------------------------------------------

    examples = (
        images
        .groupby("class_name", group_keys=False)
        .sample(n=1, random_state=42)
        .sort_values("class_name")
    )

    for _, row in examples.iterrows():

        image_name = row["image_name"]
        image_id = row["image_id"]
        class_name = row["class_name"]

        image = Image.open(row["image_path"]).convert("RGB")

        # ----------------------------------------------------
        # Create bounding-box visualization
        # ----------------------------------------------------

        boxed = image.copy()
        draw = ImageDraw.Draw(boxed)

        image_objects = objects[
            objects["image_name"] == image_name
        ]

        for _, obj in image_objects.iterrows():

            x1 = int(obj["x1"])
            y1 = int(obj["y1"])
            x2 = int(obj["x2"])
            y2 = int(obj["y2"])

            # Thicker box for visibility
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

            draw.text(
                (x1, max(0, y1 - 12)),
                str(obj["object_class"]),
                fill="red",
            )

        # ----------------------------------------------------
        # Load target mask
        # ----------------------------------------------------

        target_mask = Image.open(
            row["target_mask_path"]
        ).convert("L")

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(13, 4),
        )

        axes[0].imshow(image)
        axes[0].set_title(
            f"Original\n{class_name} | {row['lighting']}"
        )
        axes[0].axis("off")

        axes[1].imshow(boxed)
        axes[1].set_title(
            f"Bounding boxes\n"
            f"{int(row['n_valid_objects'])} objects"
        )
        axes[1].axis("off")

        axes[2].imshow(target_mask, cmap="gray")
        axes[2].set_title(
            f"Target-region mask\n"
            f"area={row['target_area_fraction']:.3f}"
        )
        axes[2].axis("off")

        plt.tight_layout()

        output_path = (
            QA_DIR
            / f"{class_name}_{image_id}_qa.png"
        )

        plt.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
