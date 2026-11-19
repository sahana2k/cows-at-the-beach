#!/usr/bin/env python3
"""Zero-shot CLIP evaluation on ImageNet-9 background variants."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


CLASS_NAMES = [
    "dog",
    "bird",
    "wheeled vehicle",
    "reptile",
    "carnivorous mammal other than a dog",
    "insect",
    "musical instrument",
    "primate",
    "fish",
]

DISPLAY_NAMES = [
    "Dog",
    "Bird",
    "Wheeled vehicle",
    "Reptile",
    "Carnivore",
    "Insect",
    "Instrument",
    "Primate",
    "Fish",
]

PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photograph of a {}.",
    "an image of a {}.",
    "a close-up photo of a {}.",
    "a natural image containing a {}.",
]

VARIANT_ALIASES = {
    "original": ["original"],
    "mixed_same": ["mixed_same", "mixed-same", "mixedsame"],
    "mixed_rand": ["mixed_rand", "mixed-rand", "mixedrand"],
    "mixed_next": ["mixed_next", "mixed-next", "mixednext"],
    "fg_mask": ["fg_mask", "fg-mask", "fgmask"],
    "only_fg": ["only_fg", "only-fg", "onlyfg"],
    "only_bg_t": ["only_bg_t", "only-bg-t", "onlybgt"],
    "only_bg_b": ["only_bg_b", "only-bg-b", "onlybgb"],
    "no_fg": ["no_fg", "no-fg", "nofg"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="original,mixed_same,mixed_rand,fg_mask",
        help="Comma-separated variants.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-name", default="RN50")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Use a balanced subset for a quick preliminary run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--assume-sorted-classes",
        action="store_true",
        help="Map sorted folders to the official IN-9 order if names are unrecognizable.",
    )
    return parser.parse_args()


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_variant_dir(root: Path, variant: str) -> Path:
    aliases = VARIANT_ALIASES.get(variant, [variant])
    candidates = []
    for alias in aliases:
        candidates.extend([root / alias / "val", root / alias, root / "val" / alias])
    for candidate in candidates:
        if candidate.is_dir() and any(item.is_dir() for item in candidate.iterdir()):
            return candidate
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find '{variant}'. Tried:\n  {tried}")


def infer_canonical_index(folder_name: str) -> int | None:
    name = normalized(folder_name)
    leading = re.match(r"^(0?[0-8])(?:[a-z]|$)", name)
    if leading:
        return int(leading.group(1))
    rules = [
        (0, ("dog",)),
        (1, ("bird",)),
        (2, ("wheel", "vehicle")),
        (3, ("reptile",)),
        (4, ("carnivore",)),
        (5, ("insect",)),
        (6, ("instrument",)),
        (7, ("primate",)),
        (8, ("fish",)),
    ]
    for index, keywords in rules:
        if any(keyword in name for keyword in keywords):
            return index
    return None


def class_remap(dataset: ImageFolder, assume_sorted: bool) -> dict[int, int]:
    if len(dataset.classes) != 9:
        raise ValueError(
            f"Expected 9 class folders, found {len(dataset.classes)}: {dataset.classes}"
        )
    mapping = {}
    for source_index, folder_name in enumerate(dataset.classes):
        canonical_index = infer_canonical_index(folder_name)
        if canonical_index is None:
            if assume_sorted:
                canonical_index = source_index
            else:
                raise ValueError(
                    f"Cannot map class folder '{folder_name}'. Rename folders with an "
                    "IN-9 index/name or rerun with --assume-sorted-classes after verifying "
                    "the official order."
                )
        mapping[source_index] = canonical_index
    if set(mapping.values()) != set(range(9)):
        raise ValueError(f"Class mapping is not one-to-one: {mapping}")
    return mapping


def balanced_subset(dataset: ImageFolder, maximum: int, seed: int) -> Subset:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {index: [] for index in range(9)}
    for index, (_, label) in enumerate(dataset.samples):
        by_class[label].append(index)
    selected = []
    for indices in by_class.values():
        rng.shuffle(indices)
        selected.extend(indices[:maximum])
    rng.shuffle(selected)
    return Subset(dataset, selected)


def build_text_features(model, tokenizer, device: torch.device) -> torch.Tensor:
    per_class = []
    with torch.inference_mode():
        for class_name in CLASS_NAMES:
            prompts = [template.format(class_name) for template in PROMPT_TEMPLATES]
            tokens = tokenizer(prompts).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            mean_feature = features.mean(dim=0)
            per_class.append(mean_feature / mean_feature.norm())
    return torch.stack(per_class)


def bootstrap_accuracy_ci(labels: np.ndarray, predictions: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    values = np.empty(1000, dtype=np.float64)
    n = len(labels)
    for index in range(len(values)):
        sample = rng.integers(0, n, size=n)
        values[index] = np.mean(labels[sample] == predictions[sample])
    return np.quantile(values, [0.025, 0.975])


def plot_confusion(matrix: np.ndarray, title: str, output_path: Path) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(normalized_matrix, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(9), DISPLAY_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(9), DISPLAY_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_variant_accuracy(summary: pd.DataFrame, output_path: Path) -> None:
    lower = summary["accuracy"] - summary["ci95_low"]
    upper = summary["ci95_high"] - summary["accuracy"]
    fig, axis = plt.subplots(figsize=(8, 5))
    palette = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#ea580c"]
    colors = [palette[index % len(palette)] for index in range(len(summary))]
    bars = axis.bar(
        summary["variant"],
        100 * summary["accuracy"],
        yerr=np.vstack([100 * lower, 100 * upper]),
        capsize=5,
        color=colors,
    )
    axis.set_ylabel("Zero-shot accuracy (%)")
    axis.set_xlabel("ImageNet-9 variant")
    axis.set_ylim(0, 100)
    axis.set_title("CLIP zero-shot accuracy under background shifts")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, summary["accuracy"]):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                  f"{100 * value:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}, pretrained weights: {args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    text_features = build_text_features(model, tokenizer, device)

    all_rows = []
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    for variant_number, variant in enumerate(variants):
        variant_dir = resolve_variant_dir(args.data_root, variant)
        dataset = ImageFolder(variant_dir, transform=preprocess)
        mapping = class_remap(dataset, args.assume_sorted_classes)
        evaluation_data = dataset
        if args.max_images_per_class is not None:
            evaluation_data = balanced_subset(
                dataset, args.max_images_per_class, args.seed
            )
        loader = DataLoader(
            evaluation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        true_labels, predictions, confidences = [], [], []
        with torch.inference_mode():
            for images, source_labels in tqdm(loader, desc=variant):
                images = images.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    image_features = model.encode_image(images)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    logits = 100.0 * image_features @ text_features.T
                probabilities = logits.float().softmax(dim=-1)
                confidence, prediction = probabilities.max(dim=-1)
                canonical_labels = torch.tensor(
                    [mapping[int(label)] for label in source_labels], dtype=torch.long
                )
                true_labels.extend(canonical_labels.tolist())
                predictions.extend(prediction.cpu().tolist())
                confidences.extend(confidence.cpu().tolist())

        labels_np = np.asarray(true_labels)
        predictions_np = np.asarray(predictions)
        accuracy = accuracy_score(labels_np, predictions_np)
        macro_f1 = f1_score(labels_np, predictions_np, average="macro")
        ci_low, ci_high = bootstrap_accuracy_ci(
            labels_np, predictions_np, args.seed + variant_number
        )
        matrix = confusion_matrix(labels_np, predictions_np, labels=range(9))

        pd.DataFrame(
            {
                "true_index": labels_np,
                "true_class": [DISPLAY_NAMES[index] for index in labels_np],
                "predicted_index": predictions_np,
                "predicted_class": [DISPLAY_NAMES[index] for index in predictions_np],
                "confidence": confidences,
            }
        ).to_csv(args.output_dir / f"predictions_{variant}.csv", index=False)
        plot_confusion(
            matrix,
            f"CLIP confusion matrix: {variant}",
            args.output_dir / f"confusion_{variant}.png",
        )
        all_rows.append(
            {
                "variant": variant,
                "n": len(labels_np),
                "accuracy": accuracy,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "macro_f1": macro_f1,
                "data_path": str(variant_dir),
            }
        )
        print(
            f"{variant}: n={len(labels_np)}, accuracy={accuracy:.4f}, "
            f"macro-F1={macro_f1:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]"
        )

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_variant_accuracy(summary, args.output_dir / "variant_accuracy.png")

    indexed = summary.set_index("variant")
    extra = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "classes": DISPLAY_NAMES,
        "prompt_templates": PROMPT_TEMPLATES,
        "seed": args.seed,
        "device": str(device),
    }
    if {"mixed_same", "mixed_rand"}.issubset(indexed.index):
        extra["bg_gap"] = float(
            indexed.loc["mixed_same", "accuracy"]
            - indexed.loc["mixed_rand", "accuracy"]
        )
        print(f"BG-Gap: {100 * extra['bg_gap']:.2f} percentage points")
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(extra, indent=2), encoding="utf-8"
    )
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
