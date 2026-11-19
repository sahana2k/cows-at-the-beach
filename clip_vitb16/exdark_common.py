"""
Shared code for the CLIP ViT-B/16 ExDark attribution stage.

Everything that must stay identical across scripts lives here:
paths, class list, prompts, model loading, CLIP preprocessing geometry,
and the attribution methods themselves.

Put this file in  ~/cse598_vlm_project/scripts/exdark/  next to the other
scripts. Every new script imports it.
"""

from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# PATHS  (same layout as prepare_exdark.py / run_clip_exdark.py)
# ============================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "EXDARK_PROJECT_ROOT",
        str(Path.home() / "cse598_vlm_project"),
    )
)

METADATA_DIR = PROJECT_ROOT / "data/exdark/processed/metadata"
IMAGE_CSV = METADATA_DIR / "exdark_images.csv"
OBJECT_CSV = METADATA_DIR / "exdark_objects.csv"

CROPPED_MASK_DIR = PROJECT_ROOT / "data/exdark/processed/masks_clip224"

OUTPUT_DIR = PROJECT_ROOT / "outputs/exdark"
PREDICTIONS_CSV = OUTPUT_DIR / "clip_vitb16_testing_predictions.csv"
ATTR_DIR = OUTPUT_DIR / "attribution"

MODEL_NAME = "openai/clip-vit-base-patch16"

IMAGE_SIZE = 224
PATCH = 16
GRID = IMAGE_SIZE // PATCH  # 14


# ============================================================
# CLASSES AND PROMPTS  (unchanged from run_clip_exdark.py)
# ============================================================

CLASS_NAMES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
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

# Optional prompt ensemble (--prompts ensemble).
# REPLACE these with the exact 5 templates the RN50 branch uses so the two
# CLIP branches are comparable. "{}" is filled with the lowercase class
# name ("people" stays "people").
ENSEMBLE_TEMPLATES = [
    "a photo of a {}.",
    "a dark photo of a {}.",
    "a low-light photo of a {}.",
    "a photo of the {} at night.",
    "a blurry photo of a {}.",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

LOGIT_SCALE = 100.0  # same fixed scale as run_clip_exdark.py


# ============================================================
# MODEL
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(output):
    """Projected-feature extraction that works across Transformers versions."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unexpected CLIP output type: {type(output)}")


def load_clip(device):
    """Return (model, processor). float32, eval mode."""
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


@torch.no_grad()
def encode_text(model, processor, device, prompts="single"):
    """(12, D) L2-normalised text features, rows ordered as CLASS_NAMES."""

    def enc(texts):
        tok = processor(text=texts, return_tensors="pt", padding=True)
        tok = {k: v.to(device) for k, v in tok.items()}
        f = _as_tensor(model.get_text_features(**tok))
        return f / f.norm(dim=-1, keepdim=True)

    if prompts == "single":
        return enc([PROMPT_MAP[c] for c in CLASS_NAMES])

    if prompts == "ensemble":
        feats = []
        for c in CLASS_NAMES:
            name = c.lower()
            f = enc([t.format(name) for t in ENSEMBLE_TEMPLATES]).mean(0)
            feats.append(f / f.norm())
        return torch.stack(feats)

    raise ValueError(prompts)


def preprocess(processor, pil_images, device):
    """Exactly the preprocessing used for the 79.7% baseline."""
    px = processor(images=pil_images, return_tensors="pt")["pixel_values"]
    return px.to(device)


def load_rgb(path):
    # Same as the baseline: no EXIF transpose (masks use the raw pixel grid too).
    return Image.open(path).convert("RGB")


# ============================================================
# MANUAL ViT FORWARD
#
# Re-implements the HF CLIP vision tower step by step using the model's own
# weights, so we can keep every block's tokens and every attention matrix in
# the autograd graph. This does not depend on output_hidden_states /
# output_attentions, which behave differently across Transformers versions.
# check_forward_matches() verifies the result equals get_image_features().
# ============================================================

def _attention(attn_mod, x):
    B, N, D = x.shape
    H = attn_mod.num_heads
    hd = D // H
    q = attn_mod.q_proj(x) * (hd ** -0.5)
    k = attn_mod.k_proj(x)
    v = attn_mod.v_proj(x)
    q = q.view(B, N, H, hd).transpose(1, 2)
    k = k.view(B, N, H, hd).transpose(1, 2)
    v = v.view(B, N, H, hd).transpose(1, 2)
    attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1)  # (B,H,N,N)
    out = (attn @ v).transpose(1, 2).reshape(B, N, D)
    return attn_mod.out_proj(out), attn


def vit_forward(model, pixel_values, keep=False):
    """
    Returns (image_features_normalised, blocks, attns).

    blocks[L] = token output of transformer block L, L = 1..12
                (blocks[0] = input to block 1). Shape (B, 197, 768).
    attns[i]  = softmax attention of block i+1. Shape (B, 12, 197, 197).
    """
    vm = model.vision_model
    x = vm.embeddings(pixel_values)
    x = vm.pre_layrnorm(x)

    blocks, attns = [x], []
    for layer in vm.encoder.layers:
        r = x
        h, a = _attention(layer.self_attn, layer.layer_norm1(x))
        x = r + h
        x = x + layer.mlp(layer.layer_norm2(x))
        blocks.append(x)
        attns.append(a)

    pooled = vm.post_layernorm(x[:, 0, :])
    f = model.visual_projection(pooled)
    f = f / f.norm(dim=-1, keepdim=True)
    return f, blocks, attns


@torch.no_grad()
def check_forward_matches(model, pixel_values):
    f_manual, _, _ = vit_forward(model, pixel_values)
    f_ref = _as_tensor(model.get_image_features(pixel_values=pixel_values))
    f_ref = f_ref / f_ref.norm(dim=-1, keepdim=True)
    err = (f_manual - f_ref).abs().max().item()
    if err > 1e-3:
        raise RuntimeError(f"Manual ViT forward differs from HF ({err:.2e}).")
    return err


# ============================================================
# ATTRIBUTION METHODS
# ============================================================

def method_names(gradcam_layers):
    names = [f"gradcam_L{L}" for L in gradcam_layers]
    names += ["chefer_full", "chefer_last"]
    return names


def compute_attributions(model, text_feats, pixel_values, class_idx,
                         gradcam_layers=(11, 10, 8)):
    """
    Attribution maps for the given class of every image in the batch.

    class_idx: LongTensor (B,), class to explain for each image.
    Returns dict  method_name -> np.ndarray (B, 14, 14), float32, >= 0,
            and logits (B, 12) as numpy.

    Batching is safe: ViT has no batch-coupled layers, so the gradient of
    sum_i logit_i wrt image i's activations equals its single-image gradient.

    Grad-CAM note: never use L=12. The prediction reads only the CLS token of
    block 12, so gradients on block-12 patch tokens are exactly zero. L=11 is
    the standard "last usable" layer (= input of the final block).
    """
    for L in gradcam_layers:
        if not 1 <= L <= 11:
            raise ValueError("Grad-CAM layers must be in 1..11 (see docstring).")

    with torch.enable_grad():
        # Weights are frozen, so the input must require grad for a graph to exist.
        f, blocks, attns = vit_forward(
            model, pixel_values.detach().clone().requires_grad_(True))
        logits = LOGIT_SCALE * f @ text_feats.T
        score = logits.gather(1, class_idx.view(-1, 1)).sum()

        targets = [blocks[L] for L in gradcam_layers] + list(attns)
        grads = torch.autograd.grad(score, targets)

    B = pixel_values.shape[0]
    out = {}

    # ---- Grad-CAM on patch tokens -------------------------------------
    for L, g in zip(gradcam_layers, grads[: len(gradcam_layers)]):
        A = blocks[L][:, 1:, :].detach()      # (B,196,D)
        G = g[:, 1:, :]
        w = G.mean(dim=1, keepdim=True)       # channel weights
        cam = F.relu((A * w).sum(-1))         # (B,196)
        out[f"gradcam_L{L}"] = cam.view(B, GRID, GRID)

    # ---- Chefer et al. 2021 (gradient-weighted attention rollout) ------
    attn_grads = grads[len(gradcam_layers):]
    N = attns[0].shape[-1]
    eye = torch.eye(N, device=pixel_values.device).expand(B, N, N)

    def chefer(start):
        R = eye.clone()
        for a, g in zip(attns[start:], attn_grads[start:]):
            cam = (g * a.detach()).clamp(min=0).mean(dim=1)  # (B,N,N)
            R = R + torch.bmm(cam, R)
        return R[:, 0, 1:].reshape(B, GRID, GRID)

    out["chefer_full"] = chefer(0)
    out["chefer_last"] = chefer(len(attns) - 1)  # official CLIP-notebook default

    out = {k: v.float().cpu().numpy() for k, v in out.items()}
    return out, logits.detach().cpu().numpy()


@torch.no_grad()
def class_logits(model, text_feats, pixel_values):
    f, _, _ = vit_forward(model, pixel_values)
    return LOGIT_SCALE * f @ text_feats.T


# ============================================================
# CLIP PREPROCESSING GEOMETRY (for masks)
#
# CLIPProcessor: resize so the SHORT side = 224, then centre-crop 224x224.
# Masks are made full-resolution, so they must go through the same geometry.
# Two HF code paths round the crop offset differently ("floor" in the PIL
# processor, "round" in the torchvision one); make_cropped_masks.py detects
# which one your installed processor uses.
# ============================================================

def resize_shape(w, h, size=IMAGE_SIZE):
    if w <= h:
        return size, int(size * h / w)
    return int(size * w / h), size


def crop_offsets(new_w, new_h, size=IMAGE_SIZE, mode="floor"):
    if mode == "floor":
        return (new_h - size) // 2, (new_w - size) // 2
    return int(round((new_h - size) / 2.0)), int(round((new_w - size) / 2.0))


def mask_to_clip_frame(mask_u8, mode="floor", size=IMAGE_SIZE):
    """
    Full-resolution 0/255 mask -> (cropped_soft_224x224 in [0,1],
                                   crop_survival in [0,1] or nan).

    Uses area-averaged (BOX) resizing so tiny objects are not lost; the
    resulting soft mask gives each pixel its exact coverage.
    """
    h, w = mask_u8.shape
    new_w, new_h = resize_shape(w, h, size)
    m = Image.fromarray(mask_u8).resize((new_w, new_h), Image.BOX)
    m = np.asarray(m, dtype=np.float32) / 255.0
    top, left = crop_offsets(new_w, new_h, size, mode)
    crop = m[top: top + size, left: left + size]
    total = m.sum()
    survival = float(crop.sum() / total) if total > 0 else float("nan")
    return crop, survival


def image_to_clip_frame(pil, mode="floor", size=IMAGE_SIZE):
    """Display-only RGB crop (uint8) in the same 224x224 frame."""
    w, h = pil.size
    new_w, new_h = resize_shape(w, h, size)
    im = pil.resize((new_w, new_h), Image.BICUBIC)
    top, left = crop_offsets(new_w, new_h, size, mode)
    return np.asarray(im)[top: top + size, left: left + size]


def boxes_to_mask(boxes, w, h):
    """boxes: iterable of (x1,y1,x2,y2) in original pixels -> 0/255 uint8."""
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[int(y1): int(y2), int(x1): int(x2)] = 255
    return m


# ============================================================
# OAL
# ============================================================

def upsample(maps_14):
    """(N,14,14) -> (N,224,224) bilinear, same as standard Grad-CAM display."""
    t = torch.from_numpy(np.asarray(maps_14, dtype=np.float32))[:, None]
    t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                      align_corners=False)
    return t[:, 0].clamp(min=0).numpy()


def oal_stats(heat_224, mask_soft):
    """
    heat_224: (224,224) >= 0;  mask_soft: (224,224) in [0,1].
    Returns (attr_frac_in, area_frac, oal). NaN when undefined.
    """
    area = float(mask_soft.mean())
    total = float(heat_224.sum())
    if area <= 0 or total <= 0:
        return float("nan"), area, float("nan")
    inside = float((heat_224 * mask_soft).sum()) / total
    return inside, area, inside / area
