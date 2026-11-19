#!/usr/bin/env python3
"""Grad-CAM and object-attribution-localization (OAL) for CLIP RN50.

The default pilot samples 10 foregrounds from each of the nine ImageNet-9
classes and evaluates the same foregrounds under original, mixed_same, and
mixed_rand backgrounds (90 images per variant; 270 forward/backward passes).

OAL is attribution enrichment inside the foreground mask:

    OAL = (Grad-CAM mass inside mask / total Grad-CAM mass) / mask area fraction

Uniform/random attribution has expected OAL 1. Values above 1 indicate that
attribution is more concentrated on the foreground than expected from its area.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from evaluate_clip import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    PROMPT_TEMPLATES,
    build_text_features,
    infer_canonical_index,
    resolve_variant_dir,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--variants", default="original,mixed_same,mixed_rand",
        help="Comma-separated RGB variants to evaluate.",
    )
    parser.add_argument("--mask-variant", default="fg_mask")
    parser.add_argument("--output-dir", type=Path, default=Path("results_oal_pilot"))
    parser.add_argument("--model-name", default="RN50-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--images-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cam-target", choices=("predicted", "true"), default="predicted",
        help="Class logit differentiated for Grad-CAM. Predicted is preferred for RQ1.",
    )
    parser.add_argument("--save-examples", type=int, default=3)
    return parser.parse_args()


def foreground_id(path: Path) -> str:
    """Return the foreground ImageNet id from original or composite filename."""
    match = re.search(r"(?:^|fg_)(n\d+_\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse foreground id from {path.name}")
    return match.group(1)


def image_files(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def indexed_variant(variant_dir: Path) -> dict[tuple[int, str], Path]:
    index: dict[tuple[int, str], Path] = {}
    for class_dir in sorted(path for path in variant_dir.iterdir() if path.is_dir()):
        class_index = infer_canonical_index(class_dir.name)
        if class_index is None:
            raise ValueError(f"Cannot map ImageNet-9 class folder: {class_dir.name}")
        for path in image_files(class_dir):
            key = (class_index, foreground_id(path))
            if key in index:
                raise ValueError(f"Duplicate foreground {key} in {variant_dir}")
            index[key] = path
    return index


def choose_pilot_keys(
    original_index: dict[tuple[int, str], Path], images_per_class: int, seed: int
) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for key in original_index:
        grouped[key[0]].append(key)
    chosen: list[tuple[int, str]] = []
    for class_index in range(9):
        candidates = sorted(grouped[class_index])
        if len(candidates) < images_per_class:
            raise ValueError(
                f"Class {class_index} has {len(candidates)} images; "
                f"cannot sample {images_per_class}."
            )
        rng.shuffle(candidates)
        chosen.extend(candidates[:images_per_class])
    return chosen


def mask_path_for(mask_dir: Path, class_index: int, fg_id: str) -> Path:
    matching_classes = [
        path for path in mask_dir.iterdir()
        if path.is_dir() and infer_canonical_index(path.name) == class_index
    ]
    if len(matching_classes) != 1:
        raise FileNotFoundError(
            f"Expected one mask folder for class {class_index}; found {matching_classes}"
        )
    path = matching_classes[0] / f"{fg_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing foreground mask: {path}")
    return path


def square_size(model) -> int:
    size = getattr(model.visual, "image_size", 224)
    if isinstance(size, (tuple, list)):
        if size[0] != size[1]:
            raise ValueError(f"Expected a square CLIP input, got {size}")
        return int(size[0])
    return int(size)


def transformed_mask(mask_path: Path, size: int) -> torch.Tensor:
    mask = np.load(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    mask_image = TF.resize(mask_image, size, interpolation=InterpolationMode.NEAREST)
    mask_image = TF.center_crop(mask_image, [size, size])
    return torch.from_numpy(np.asarray(mask_image) > 0)


class GradCamCapture:
    def __init__(self, layer: torch.nn.Module):
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("The selected Grad-CAM layer did not return a tensor.")
        self.activation = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: torch.Tensor):
        self.gradient = gradient

    def heatmap(self, output_size: int) -> torch.Tensor:
        if self.activation is None or self.gradient is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(output_size, output_size), mode="bilinear", align_corners=False
        )[0, 0]
        maximum = cam.max()
        if maximum > 0:
            cam = cam / maximum
        return cam.detach().cpu()

    def close(self):
        self.handle.remove()


def oal_score(cam: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
    cam_np = cam.numpy().astype(np.float64)
    mask_np = mask.numpy().astype(bool)
    total = cam_np.sum()
    area_fraction = float(mask_np.mean())
    if total <= 0 or area_fraction <= 0:
        return float("nan"), area_fraction, float("nan")
    inside_fraction = float(cam_np[mask_np].sum() / total)
    return inside_fraction / area_fraction, area_fraction, inside_fraction


def add_overlay_example(
    examples: list[dict], image_path: Path, cam: torch.Tensor, mask: torch.Tensor,
    row: dict,
) -> None:
    image = Image.open(image_path).convert("RGB").resize((cam.shape[1], cam.shape[0]))
    examples.append(
        {"image": np.asarray(image), "cam": cam.numpy(), "mask": mask.numpy(), **row}
    )


def plot_examples(examples: list[dict], output_path: Path) -> None:
    if not examples:
        return
    fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 4.3))
    axes = np.atleast_1d(axes)
    for axis, item in zip(axes, examples):
        axis.imshow(item["image"])
        axis.imshow(item["cam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axis.contour(item["mask"], levels=[0.5], colors="white", linewidths=1.2)
        status = "correct" if item["correct"] else "wrong"
        axis.set_title(
            f"{item['variant']} | {status}\n"
            f"pred: {item['predicted_class']} | OAL {item['oal']:.2f}",
            fontsize=10,
        )
        axis.axis("off")
    fig.suptitle("CLIP RN50 Grad-CAM (white outline = foreground mask)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["variant", "correct"], sort=False)["oal"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "mean_oal", "std": "std_oal"})
    )
    summary["sem_oal"] = summary["std_oal"] / np.sqrt(summary["n"])
    return summary


def plot_summary(summary: pd.DataFrame, variants: list[str], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    width = 0.34
    for offset, correct, label, color in [
        (-width / 2, True, "Correct predictions", "#168391"),
        (width / 2, False, "Incorrect predictions", "#e65313"),
    ]:
        means, errors = [], []
        for variant in variants:
            match = summary[(summary["variant"] == variant) & (summary["correct"] == correct)]
            means.append(float(match["mean_oal"].iloc[0]) if len(match) else np.nan)
            errors.append(float(match["sem_oal"].iloc[0]) if len(match) else 0.0)
        bars = axis.bar(x + offset, means, width, yerr=errors, capsize=4,
                        label=label, color=color)
        for bar, value in zip(bars, means):
            if np.isfinite(value):
                axis.text(bar.get_x() + bar.get_width()/2, value + 0.04,
                          f"{value:.2f}", ha="center", fontsize=9)
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1,
                 label="Uniform-attribution baseline")
    axis.set_xticks(x, [name.replace("_", "-").title() for name in variants])
    axis.set_ylabel("Mean OAL (± SEM)")
    axis.set_title("CLIP RN50 object attribution by prediction correctness")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}, pretrained: {args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text_features = build_text_features(model, tokenizer, device).detach()
    input_size = square_size(model)

    if not hasattr(model.visual, "layer4"):
        raise ValueError(
            "This script expects a ResNet CLIP visual encoder with visual.layer4. "
            f"Received {type(model.visual).__name__}."
        )
    capture = GradCamCapture(model.visual.layer4)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    variant_dirs = {name: resolve_variant_dir(args.data_root, name) for name in variants}
    if "original" not in variant_dirs:
        raise ValueError("Include original in --variants so a shared pilot subset can be chosen.")
    mask_dir = resolve_variant_dir(args.data_root, args.mask_variant)
    indices = {name: indexed_variant(path) for name, path in variant_dirs.items()}
    keys = choose_pilot_keys(indices["original"], args.images_per_class, args.seed)

    missing = {
        variant: [key for key in keys if key not in indices[variant]]
        for variant in variants
    }
    missing = {variant: values for variant, values in missing.items() if values}
    if missing:
        raise FileNotFoundError(f"Selected foregrounds are not shared across variants: {missing}")

    rows: list[dict] = []
    examples: list[dict] = []
    # Use one foreground under every background so the example panel mirrors
    # the paired comparison in the Backgrounds Challenge.
    example_key = keys[0] if args.save_examples > 0 else None
    start = time.perf_counter()
    try:
        for variant in variants:
            for true_index, fg_id in tqdm(keys, desc=f"Grad-CAM {variant}"):
                image_path = indices[variant][(true_index, fg_id)]
                mask_path = mask_path_for(mask_dir, true_index, fg_id)
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                image_tensor.requires_grad_(True)

                capture.activation = None
                capture.gradient = None
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ text_features.T
                probabilities = logits.detach().float().softmax(dim=-1)
                confidence, predicted = probabilities[0].max(dim=-1)
                predicted_index = int(predicted)
                target_index = predicted_index if args.cam_target == "predicted" else true_index
                logits[0, target_index].backward()

                cam = capture.heatmap(input_size)
                mask = transformed_mask(mask_path, input_size)
                oal, area_fraction, inside_fraction = oal_score(cam, mask)
                row = {
                    "variant": variant,
                    "foreground_id": fg_id,
                    "true_index": true_index,
                    "true_class": DISPLAY_NAMES[true_index],
                    "predicted_index": predicted_index,
                    "predicted_class": DISPLAY_NAMES[predicted_index],
                    "correct": predicted_index == true_index,
                    "confidence": float(confidence),
                    "cam_target": args.cam_target,
                    "oal": oal,
                    "object_area_fraction": area_fraction,
                    "attribution_inside_fraction": inside_fraction,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
                rows.append(row)
                if (true_index, fg_id) == example_key:
                    add_overlay_example(examples, image_path, cam, mask, row)
                del image_tensor, image_features, logits
    finally:
        capture.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "oal_predictions.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output_dir / "oal_summary.csv", index=False)
    plot_summary(summary, variants, args.output_dir / "oal_by_correctness.png")
    plot_examples(examples, args.output_dir / "gradcam_examples.png")

    elapsed = time.perf_counter() - start
    metadata = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "variants": variants,
        "images_per_class": args.images_per_class,
        "total_forward_backward_passes": len(frame),
        "cam_target": args.cam_target,
        "gradcam_layer": "visual.layer4",
        "oal_definition": "attribution mass fraction in mask / mask area fraction",
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "class_names": CLASS_NAMES,
        "prompt_templates": PROMPT_TEMPLATES,
    }
    (args.output_dir / "oal_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Completed {len(frame)} passes in {elapsed:.1f} seconds")
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
