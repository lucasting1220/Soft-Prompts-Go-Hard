"""
Differentiable LLaVA-NeXT (anyres) image preprocessing.

The HuggingFace `LlavaNextImageProcessor` builds `pixel_values` with numpy / PIL
ops (resize, pad, tile, center-crop, normalize) that are NOT differentiable, so
you cannot backprop from the model loss to the raw image pixels through it.

This module reproduces the SAME pipeline with torch ops so gradients flow back to
a raw [0,1] image tensor. It reuses the exact integer-geometry helpers from
transformers (`select_best_resolution`, `get_patch_output_size`) so the tiling
grid matches the real processor; only the pixel interpolation differs (torch
bicubic vs PIL bicubic).

Because the interpolation is not bit-identical to PIL, ALWAYS run
`sanity_check_preprocess` once before trusting an attack: it reports the max/mean
abs difference between this function's `pixel_values` and the real processor's.
If the gap is tiny, the attack optimized here transfers to real inference.
"""
import numpy as np
import torch
import torch.nn.functional as F

from transformers.image_processing_utils import select_best_resolution, get_patch_output_size
from transformers.image_utils import ChannelDimension


def _read_processor_config(processor):
    ip = processor.image_processor
    size = ip.size
    shortest = size.get("shortest_edge", size.get("height"))
    crop = ip.crop_size["height"]
    cfg = {
        "shortest_edge": int(shortest),
        "crop": int(crop),
        "pinpoints": [list(p) for p in ip.image_grid_pinpoints],
        "mean": torch.tensor(ip.image_mean, dtype=torch.float32),
        "std": torch.tensor(ip.image_std, dtype=torch.float32),
        "do_resize": ip.do_resize,
        "do_center_crop": ip.do_center_crop,
        "do_normalize": ip.do_normalize,
    }
    return cfg


def _resize01(img, new_h, new_w):
    """img: (3,H,W) float in [0,1]. Bicubic resize with antialias to match PIL better."""
    out = F.interpolate(
        img.unsqueeze(0), size=(int(new_h), int(new_w)),
        mode="bicubic", align_corners=False, antialias=True,
    ).squeeze(0)
    return out.clamp(0, 1)


def _center_crop(img, crop):
    _, h, w = img.shape
    top = max((h - crop) // 2, 0)
    left = max((w - crop) // 2, 0)
    cropped = img[:, top:top + crop, left:left + crop]
    # if smaller than crop (shouldn't happen after resize), pad to crop
    ch, cw = cropped.shape[1], cropped.shape[2]
    if ch != crop or cw != crop:
        cropped = F.pad(cropped, (0, crop - cw, 0, crop - ch), value=0)
    return cropped


def _base_shortest_edge_size(h, w, shortest):
    """Replicate CLIP-style shortest-edge resize target (h,w), aspect preserved."""
    if h <= w:
        new_h = shortest
        new_w = int(round(shortest * w / h))
    else:
        new_w = shortest
        new_h = int(round(shortest * h / w))
    return new_h, new_w


def build_pixel_values(img01, processor, device=None, dtype=None):
    """
    img01: (3, H, W) float tensor in [0,1] at the ORIGINAL image resolution.
    Returns:
        pixel_values: (1, num_patches, 3, crop, crop)
        image_sizes:  (1, 2) long tensor = [[H, W]]
    Fully differentiable w.r.t. img01.
    """
    cfg = _read_processor_config(processor)
    crop = cfg["crop"]
    shortest = cfg["shortest_edge"]
    mean = cfg["mean"].to(img01.device).view(3, 1, 1)
    std = cfg["std"].to(img01.device).view(3, 1, 1)

    _, H, W = img01.shape

    # --- anyres tiles ---
    best_h, best_w = select_best_resolution((H, W), cfg["pinpoints"])
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    new_h, new_w = get_patch_output_size(dummy, (best_h, best_w), ChannelDimension.LAST)

    resized = _resize01(img01, new_h, new_w)

    # centered pad to (best_h, best_w), matching _get_padding_size
    paste_y, r_y = divmod(best_h - new_h, 2)
    paste_x, r_x = divmod(best_w - new_w, 2)
    padded = F.pad(resized, (paste_x, paste_x + r_x, paste_y, paste_y + r_y), value=0.0)

    tiles = []
    for i in range(0, best_h, crop):
        for j in range(0, best_w, crop):
            tiles.append(padded[:, i:i + crop, j:j + crop])

    # --- base (resized original) ---
    # The HF LlavaNext processor produces the base "resized_original_image" by
    # resizing the whole image directly to a square crop x crop (aspect-distorted),
    # NOT shortest-edge + center-crop. Verified empirically against the real
    # processor (see sanity_check_preprocess): squash matches to <0.003 mean abs,
    # whereas shortest-edge+crop is off by ~0.12.
    base = _resize01(img01, crop, crop)

    patches = [base] + tiles  # order matches HF: [resized_original] + patches

    normed = [(p - mean) / std if cfg["do_normalize"] else p for p in patches]
    pixel_values = torch.stack(normed, dim=0).unsqueeze(0)  # (1, num_patches, 3, crop, crop)

    if dtype is not None:
        pixel_values = pixel_values.to(dtype)
    if device is not None:
        pixel_values = pixel_values.to(device)

    image_sizes = torch.tensor([[H, W]], dtype=torch.long,
                               device=pixel_values.device)
    return pixel_values, image_sizes


@torch.no_grad()
def sanity_check_preprocess(pil_image, processor):
    """
    Compare our differentiable pixel_values against the real HF processor output
    for the same image. Returns (max_abs_diff, mean_abs_diff, our_shape, ref_shape).
    Run this once; if the diffs are small (e.g. < ~0.05 max), the attack is valid.
    """
    from torchvision.transforms.functional import to_tensor
    img01 = to_tensor(pil_image.convert("RGB")).float()  # (3,H,W) in [0,1]

    ours, our_sizes = build_pixel_values(img01, processor)

    ref = processor(images=pil_image, text="<image>", return_tensors="pt")
    ref_pv = ref["pixel_values"].float()

    if ref_pv.shape != ours.shape:
        return {
            "match_shape": False,
            "our_shape": tuple(ours.shape),
            "ref_shape": tuple(ref_pv.shape),
            "note": "shape mismatch -> tiling grid differs; fix before attacking",
        }

    diff = (ours - ref_pv).abs().flatten().float()
    return {
        "match_shape": True,
        "max_abs_diff": diff.max().item(),
        "p99_abs_diff": torch.quantile(diff, 0.99).item(),
        "mean_abs_diff": diff.mean().item(),
        "frac_over_0p05": (diff > 0.05).float().mean().item(),
        "our_shape": tuple(ours.shape),
        "ref_shape": tuple(ref_pv.shape),
    }


def build_region_mask(H, W, args, device="cpu"):
    """
    Build a (1,H,W) mask (1 where perturbation is allowed) on the ORIGINAL image.
    Modes:
      margin : frame around all four edges, band width round(frac*H) (top/bottom)
               and round(frac*W) (left/right)
      bottom : full-width strip of height round(frac*H) at the bottom
      corner : square of side round(frac*min(H,W)) in bottom-right corner
      box    : explicit rectangle from --region_x/y/w/h (pixels)
    """
    mask = torch.zeros(1, H, W, device=device)
    mode = args.region_mode
    if mode == "box":
        x, y = args.region_x, args.region_y
        w = args.region_w if args.region_w > 0 else W - x
        h = args.region_h if args.region_h > 0 else H - y
        mask[:, y:y + h, x:x + w] = 1.0
    elif mode == "corner":
        s = max(int(round(args.region_frac * min(H, W))), 1)
        mask[:, H - s:H, W - s:W] = 1.0
    elif mode == "bottom":
        s = max(int(round(args.region_frac * H)), 1)
        mask[:, H - s:H, :] = 1.0
    else:  # margin
        th = max(int(round(args.region_frac * H)), 1)
        tw = max(int(round(args.region_frac * W)), 1)
        mask[:, :th, :] = 1.0
        mask[:, H - th:H, :] = 1.0
        mask[:, :, :tw] = 1.0
        mask[:, :, W - tw:W] = 1.0
    return mask
