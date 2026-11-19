from pathlib import Path
import argparse

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.home() / "cse598_vlm_project"

METADATA_PATH = (
    PROJECT_ROOT
    / "data/exdark/processed/metadata/exdark_images.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs/exdark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch16"


# Keep this mapping fixed for the preliminary experiment.
CLASS_NAMES = [
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


PROMPT_MAP = {
    "Bicycle": "a photo of a bicycle",
    "Boat": "a photo of a boat",
    "Bottle": "a photo of a bottle",
    "Bus": "a photo of a bus",
    "Car": "a photo of a car",
    "Cat": "a photo of a cat",
    "Chair": "a photo of a chair",
    "Cup": "a photo of a cup",
    "Dog": "a photo of a dog",
    "Motorbike": "a photo of a motorbike",
    "People": "a photo of people",
    "Table": "a photo of a table",
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="Testing",
        choices=[
            "Training",
            "Validation",
            "Testing",
            "all",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images for debugging.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    return parser.parse_args()


def load_images(paths):

    images = []

    for path in paths:
        image = Image.open(path).convert("RGB")
        images.append(image)

    return images

def extract_features(output):
    """
    Handle Hugging Face CLIP outputs across Transformers versions.

    Some versions return the projected embedding tensor directly.
    Others may return a structured model output containing pooler_output.
    """

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]

    raise TypeError(
        f"Unexpected CLIP feature output type: {type(output)}"
    )
    
def extract_clip_features(output):
    """
    Extract projected CLIP embeddings across Transformers API versions.
    """

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    raise TypeError(
        f"Unexpected CLIP output type: {type(output)}"
    )

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("CLIP ViT-B/16 ExDark zero-shot evaluation")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    df = pd.read_csv(METADATA_PATH)

    if args.split != "all":
        df = df[df["split"] == args.split].copy()

    df = df.reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("Split:", args.split)
    print("Images:", len(df))

    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    print()
    print("Loading CLIP...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    print("CLIP loaded.")

    # --------------------------------------------------------
    # Encode class prompts ONCE
    # --------------------------------------------------------

    prompts = [
        PROMPT_MAP[class_name]
        for class_name in CLASS_NAMES
    ]

    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
    )

    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    with torch.no_grad():

        text_output = model.get_text_features(
            **text_inputs
        )
    
        text_features = extract_clip_features(
            text_output
        )
    
        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True,
            )
        )
    
    print(
        "Text feature shape:",
        tuple(text_features.shape)
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    results = []

    batch_size = args.batch_size

    for start in tqdm(
        range(0, len(df), batch_size),
        desc="CLIP inference",
    ):

        end = min(
            start + batch_size,
            len(df),
        )

        batch_df = df.iloc[start:end]

        images = load_images(
            batch_df["image_path"].tolist()
        )

        image_inputs = processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = (
            image_inputs["pixel_values"]
            .to(device)
        )

        with torch.no_grad():

            image_output = model.get_image_features(
                pixel_values=pixel_values
            )
            
            image_features = extract_clip_features(
                image_output
            )
            
            image_features = (
                image_features
                / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )
        
            similarities = (
                image_features
                @ text_features.T
            )
        
            probabilities = torch.softmax(
                similarities * 100.0,
                dim=-1,
            )

        probabilities = (
            probabilities
            .cpu()
            .numpy()
        )

        for row_index, (_, row) in enumerate(
            batch_df.iterrows()
        ):

            scores = probabilities[row_index]

            prediction_index = int(
                scores.argmax()
            )

            prediction = CLASS_NAMES[
                prediction_index
            ]

            confidence = float(
                scores[prediction_index]
            )

            result = {
                "image_name": row["image_name"],
                "image_id": row["image_id"],
                "ground_truth": row["class_name"],
                "prediction": prediction,
                "correct": (
                    prediction
                    == row["class_name"]
                ),
                "confidence": confidence,
                "lighting": row["lighting"],
                "environment": row["environment"],
                "split": row["split"],
                "target_area_fraction":
                    row["target_area_fraction"],
            }

            # Save all 12 class probabilities
            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):

                result[
                    f"prob_{class_name.lower()}"
                ] = float(
                    scores[class_index]
                )

            results.append(result)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    if args.output is None:

        split_name = args.split.lower()

        if args.limit is not None:
            filename = (
                f"clip_vitb16_"
                f"{split_name}_"
                f"limit{args.limit}.csv"
            )

        else:
            filename = (
                f"clip_vitb16_"
                f"{split_name}_predictions.csv"
            )

        output_path = (
            OUTPUT_DIR / filename
        )

    else:
        output_path = Path(args.output)

    result_df.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    accuracy = (
        result_df["correct"].mean()
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Images:   {len(result_df)}"
    )

    print(
        f"Correct:  "
        f"{int(result_df['correct'].sum())}"
    )

    print(
        f"Accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print()
    print("Predictions saved to:")
    print(output_path)

    print("=" * 70)


if __name__ == "__main__":
    main()
