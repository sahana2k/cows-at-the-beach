from pathlib import Path
import csv
import json
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.home() / "cse598_vlm_project"

EXDARK_ROOT = PROJECT_ROOT / "data/exdark"

IMAGE_ROOT = EXDARK_ROOT / "raw/images/ExDark"
ANNOTATION_ROOT = EXDARK_ROOT / "raw/annotations/ExDark_Annno"

CLASSLIST_PATH = (
    EXDARK_ROOT
    / "raw/Exclusively-Dark-Image-Dataset/Groundtruth/imageclasslist.txt"
)

PROCESSED_ROOT = EXDARK_ROOT / "processed"

METADATA_DIR = PROCESSED_ROOT / "metadata"
TARGET_MASK_DIR = PROCESSED_ROOT / "masks/target"
ALL_OBJECT_MASK_DIR = PROCESSED_ROOT / "masks/all_objects"

OUTPUT_DIR = PROJECT_ROOT / "outputs/exdark"


# ============================================================
# EXDARK LABEL DEFINITIONS
# ============================================================

CLASS_MAP = {
    1: "Bicycle",
    2: "Boat",
    3: "Bottle",
    4: "Bus",
    5: "Car",
    6: "Cat",
    7: "Chair",
    8: "Cup",
    9: "Dog",
    10: "Motorbike",
    11: "People",
    12: "Table",
}

LIGHT_MAP = {
    1: "Low",
    2: "Ambient",
    3: "Object",
    4: "Single",
    5: "Weak",
    6: "Strong",
    7: "Screen",
    8: "Window",
    9: "Shadow",
    10: "Twilight",
}

ENVIRONMENT_MAP = {
    1: "Indoor",
    2: "Outdoor",
}

SPLIT_MAP = {
    1: "Training",
    2: "Validation",
    3: "Testing",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_directories():
    """Create all processed/output directories."""
    for directory in [
        METADATA_DIR,
        TARGET_MASK_DIR,
        ALL_OBJECT_MASK_DIR,
        OUTPUT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_image_metadata():
    """
    Parse imageclasslist.txt.

    Format:
        image_name class_id light_id environment_id split_id
    """

    records = []

    with open(CLASSLIST_PATH, "r") as f:
        lines = f.readlines()

    # First line is the header
    for line_number, line in enumerate(lines[1:], start=2):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Unexpected metadata format at line {line_number}: {line}"
            )

        image_name = parts[0]
        class_id = int(parts[1])
        light_id = int(parts[2])
        environment_id = int(parts[3])
        split_id = int(parts[4])

        records.append(
            {
                "image_name": image_name,
                "image_id": Path(image_name).stem,
                "class_id": class_id,
                "class_name": CLASS_MAP[class_id],
                "light_id": light_id,
                "lighting": LIGHT_MAP[light_id],
                "environment_id": environment_id,
                "environment": ENVIRONMENT_MAP[environment_id],
                "split_id": split_id,
                "split": SPLIT_MAP[split_id],
            }
        )

    return records


def parse_annotation(annotation_path):
    """
    Parse one ExDark annotation file.

    Example:

        % bbGt version=3
        Bottle 131 88 207 556 0 0 0 0 0 0 0
        Bottle 16 3 105 625 0 0 0 0 0 0 0

    Relevant fields:
        class left top width height
    """

    objects = []

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Skip annotation-tool header
        if line.startswith("%"):
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        object_class = parts[0]

        try:
            left = int(float(parts[1]))
            top = int(float(parts[2]))
            width = int(float(parts[3]))
            height = int(float(parts[4]))
        except ValueError:
            continue

        objects.append(
            {
                "object_class": object_class,
                "left": left,
                "top": top,
                "bbox_width": width,
                "bbox_height": height,
            }
        )

    return objects


def find_annotation(class_name, image_name):
    """
    Find the annotation corresponding to an ExDark image.

    Annotation filenames normally preserve the image filename,
    but a small number may differ in extension capitalization
    (e.g. .JPG vs .jpg).
    """

    class_dir = ANNOTATION_ROOT / class_name

    # First try exact match
    exact_path = class_dir / f"{image_name}.txt"

    if exact_path.exists():
        return exact_path

    # Fall back to case-insensitive filename matching
    target_name = f"{image_name}.txt".lower()

    for candidate in class_dir.glob("*.txt"):
        if candidate.name.lower() == target_name:
            return candidate

    # Return expected path so downstream validation can
    # still report genuinely missing annotations
    return exact_path


def clip_bbox(left, top, width, height, image_width, image_height):
    """
    Clip a bounding box to image boundaries.

    Returns:
        x1, y1, x2, y2, was_clipped
    """

    raw_x1 = left
    raw_y1 = top
    raw_x2 = left + width
    raw_y2 = top + height

    x1 = max(0, min(raw_x1, image_width))
    y1 = max(0, min(raw_y1, image_height))

    x2 = max(0, min(raw_x2, image_width))
    y2 = max(0, min(raw_y2, image_height))

    was_clipped = (
        x1 != raw_x1
        or y1 != raw_y1
        or x2 != raw_x2
        or y2 != raw_y2
    )

    return x1, y1, x2, y2, was_clipped


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():

    ensure_directories()

    print("=" * 70)
    print("ExDark full-dataset preprocessing")
    print("=" * 70)

    metadata_records = load_image_metadata()

    print(f"Metadata records loaded: {len(metadata_records)}")

    if len(metadata_records) != 7363:
        print(
            f"WARNING: Expected 7363 metadata rows, "
            f"found {len(metadata_records)}"
        )

    image_rows = []
    object_rows = []
    validation_errors = []

    split_counter = Counter()
    class_counter = Counter()
    lighting_counter = Counter()

    missing_images = 0
    missing_annotations = 0
    corrupt_images = 0
    clipped_boxes = 0
    invalid_boxes = 0
    masks_generated = 0

    for record in tqdm(
        metadata_records,
        desc="Processing ExDark",
        unit="image",
    ):

        image_name = record["image_name"]
        image_id = record["image_id"]
        class_name = record["class_name"]

        image_path = IMAGE_ROOT / class_name / image_name
        annotation_path = find_annotation(
            class_name,
            image_name,
        )

        # ----------------------------------------------------
        # Validate image existence
        # ----------------------------------------------------

        if not image_path.exists():
            missing_images += 1

            validation_errors.append(
                {
                    "image_name": image_name,
                    "error_type": "missing_image",
                    "details": str(image_path),
                }
            )

            continue

        # ----------------------------------------------------
        # Read image dimensions
        # ----------------------------------------------------

        try:
            with Image.open(image_path) as image:
                image.load()
                image_width, image_height = image.size

        except Exception as exc:
            corrupt_images += 1

            validation_errors.append(
                {
                    "image_name": image_name,
                    "error_type": "corrupt_image",
                    "details": str(exc),
                }
            )

            continue

        # ----------------------------------------------------
        # Validate annotation
        # ----------------------------------------------------

        if not annotation_path.exists():
            missing_annotations += 1

            validation_errors.append(
                {
                    "image_name": image_name,
                    "error_type": "missing_annotation",
                    "details": str(annotation_path),
                }
            )

            objects = []

        else:
            objects = parse_annotation(annotation_path)

        # ----------------------------------------------------
        # Prepare masks
        # ----------------------------------------------------

        target_mask = np.zeros(
            (image_height, image_width),
            dtype=np.uint8,
        )

        all_object_mask = np.zeros(
            (image_height, image_width),
            dtype=np.uint8,
        )

        target_object_count = 0
        valid_object_count = 0

        # ----------------------------------------------------
        # Process each annotated object
        # ----------------------------------------------------

        for object_index, obj in enumerate(objects):

            left = obj["left"]
            top = obj["top"]
            bbox_width = obj["bbox_width"]
            bbox_height = obj["bbox_height"]

            if bbox_width <= 0 or bbox_height <= 0:
                invalid_boxes += 1

                validation_errors.append(
                    {
                        "image_name": image_name,
                        "error_type": "invalid_bbox",
                        "details": (
                            f"{obj['object_class']} "
                            f"{left} {top} "
                            f"{bbox_width} {bbox_height}"
                        ),
                    }
                )

                continue

            (
                x1,
                y1,
                x2,
                y2,
                was_clipped,
            ) = clip_bbox(
                left,
                top,
                bbox_width,
                bbox_height,
                image_width,
                image_height,
            )

            if was_clipped:
                clipped_boxes += 1

            # Box becomes empty after clipping
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1

                validation_errors.append(
                    {
                        "image_name": image_name,
                        "error_type": "empty_bbox_after_clipping",
                        "details": (
                            f"{obj['object_class']} "
                            f"{left} {top} "
                            f"{bbox_width} {bbox_height}"
                        ),
                    }
                )

                continue

            valid_object_count += 1

            is_target_class = (
                obj["object_class"].lower()
                == class_name.lower()
            )

            if is_target_class:
                target_object_count += 1

            # Fill all-object mask
            all_object_mask[y1:y2, x1:x2] = 255

            # Fill target-class mask only
            if is_target_class:
                target_mask[y1:y2, x1:x2] = 255

            object_rows.append(
                {
                    "image_name": image_name,
                    "image_id": image_id,
                    "image_class": class_name,
                    "object_index": object_index,
                    "object_class": obj["object_class"],
                    "is_target_class": is_target_class,
                    "original_left": left,
                    "original_top": top,
                    "original_width": bbox_width,
                    "original_height": bbox_height,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "clipped": was_clipped,
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )

        # ----------------------------------------------------
        # Save masks
        # ----------------------------------------------------

        target_mask_path = (
            TARGET_MASK_DIR
            / f"{image_id}.png"
        )

        all_object_mask_path = (
            ALL_OBJECT_MASK_DIR
            / f"{image_id}.png"
        )

        Image.fromarray(target_mask).save(target_mask_path)
        Image.fromarray(all_object_mask).save(all_object_mask_path)

        masks_generated += 1

        # ----------------------------------------------------
        # Calculate object-region statistics
        # ----------------------------------------------------

        total_pixels = image_width * image_height

        target_mask_pixels = int(
            np.count_nonzero(target_mask)
        )

        all_object_mask_pixels = int(
            np.count_nonzero(all_object_mask)
        )

        target_area_fraction = (
            target_mask_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        all_object_area_fraction = (
            all_object_mask_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Add image-level record
        # ----------------------------------------------------

        image_rows.append(
            {
                **record,
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "image_width": image_width,
                "image_height": image_height,
                "total_pixels": total_pixels,
                "n_annotated_objects": len(objects),
                "n_valid_objects": valid_object_count,
                "n_target_objects": target_object_count,
                "target_mask_path": str(target_mask_path),
                "all_object_mask_path": str(all_object_mask_path),
                "target_mask_pixels": target_mask_pixels,
                "target_area_fraction": target_area_fraction,
                "all_object_mask_pixels": all_object_mask_pixels,
                "all_object_area_fraction": all_object_area_fraction,
            }
        )

        class_counter[class_name] += 1
        split_counter[record["split"]] += 1
        lighting_counter[record["lighting"]] += 1

    # ========================================================
    # SAVE OUTPUT TABLES
    # ========================================================

    image_df = pd.DataFrame(image_rows)
    object_df = pd.DataFrame(object_rows)
    error_df = pd.DataFrame(validation_errors)

    image_csv = METADATA_DIR / "exdark_images.csv"
    object_csv = METADATA_DIR / "exdark_objects.csv"
    error_csv = METADATA_DIR / "validation_errors.csv"

    image_df.to_csv(image_csv, index=False)
    object_df.to_csv(object_csv, index=False)

    if len(error_df) > 0:
        error_df.to_csv(error_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "image_name",
                "error_type",
                "details",
            ]
        ).to_csv(error_csv, index=False)

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "expected_images": 7363,
        "metadata_records": len(metadata_records),
        "images_processed": len(image_rows),
        "objects_processed": len(object_rows),
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "corrupt_images": corrupt_images,
        "clipped_boxes": clipped_boxes,
        "invalid_boxes": invalid_boxes,
        "masks_generated": masks_generated,
        "class_counts": dict(class_counter),
        "split_counts": dict(split_counter),
        "lighting_counts": dict(lighting_counter),
    }

    summary_path = OUTPUT_DIR / "dataset_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Additional summary CSVs

    (
        image_df["class_name"]
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="count")
        .to_csv(
            OUTPUT_DIR / "class_distribution.csv",
            index=False,
        )
    )

    (
        image_df["lighting"]
        .value_counts()
        .rename_axis("lighting")
        .reset_index(name="count")
        .to_csv(
            OUTPUT_DIR / "lighting_distribution.csv",
            index=False,
        )
    )

    (
        image_df["split"]
        .value_counts()
        .rename_axis("split")
        .reset_index(name="count")
        .to_csv(
            OUTPUT_DIR / "split_distribution.csv",
            index=False,
        )
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Metadata records:       {len(metadata_records)}")
    print(f"Images processed:       {len(image_rows)}")
    print(f"Objects processed:      {len(object_rows)}")
    print(f"Masks generated:        {masks_generated}")

    print()
    print("Validation:")
    print(f"Missing images:         {missing_images}")
    print(f"Missing annotations:    {missing_annotations}")
    print(f"Corrupt images:         {corrupt_images}")
    print(f"Clipped boxes:          {clipped_boxes}")
    print(f"Invalid boxes:          {invalid_boxes}")

    print()
    print("Splits:")
    for key, value in sorted(split_counter.items()):
        print(f"  {key:<12} {value}")

    print()
    print("Classes:")
    for key, value in sorted(class_counter.items()):
        print(f"  {key:<12} {value}")

    print()
    print("Lighting:")
    for key, value in sorted(lighting_counter.items()):
        print(f"  {key:<12} {value}")

    print()
    print("Outputs:")
    print(f"  {image_csv}")
    print(f"  {object_csv}")
    print(f"  {error_csv}")
    print(f"  {summary_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
