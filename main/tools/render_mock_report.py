#!/usr/bin/env python3
"""
Render quick evidence PNGs for mock pipeline:
- CT middle slice with mask overlay (pre/post)
- best_plan.json rendered as an image
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _normalize_ct(ct: np.ndarray) -> np.ndarray:
    # Window/level for liver-ish CT
    vmin, vmax = -175.0, 600.0
    ct = np.clip(ct, vmin, vmax)
    ct = (ct - vmin) / (vmax - vmin + 1e-6)
    return ct


def _pick_mid_slice(vol: np.ndarray) -> np.ndarray:
    z = vol.shape[0] // 2
    return vol[z]


def _render_overlay(ct_slice: np.ndarray, mask_slice: np.ndarray, out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct_norm = _normalize_ct(ct_slice)
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(ct_norm, cmap="gray")

    # overlay: organ=1 (green), tumor=2 (red)
    organ = mask_slice == 1
    tumor = mask_slice == 2
    overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    overlay[organ] = [0.0, 1.0, 0.0, 0.25]
    overlay[tumor] = [1.0, 0.0, 0.0, 0.45]
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _render_text_png(text: str, out_png: Path, title: str = "Output"):
    from PIL import Image, ImageDraw, ImageFont

    lines = [title, "-" * 40] + text.strip().splitlines()
    font = ImageFont.load_default()
    line_h = 14
    width = max(len(line) for line in lines) * 7 + 20
    height = (len(lines) + 1) * line_h + 20
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="main/work/mock_data")
    ap.add_argument("--case_id", type=str, default="HCC_MOCK_001")
    ap.add_argument("--out_dir", type=str, default="main/work/screens")
    ap.add_argument("--best_plan_json", type=str, default="main/work/mock_outputs/best_plan.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    case_dir = data_root / "hcc" / args.case_id

    pre_ct = case_dir / "pre_ct.nii.gz"
    pre_mask = case_dir / "pre_mask.nii.gz"
    post_ct = case_dir / "post_ct.nii.gz"
    post_mask = case_dir / "post_mask.nii.gz"

    import nibabel as nib

    pre_ct_img = nib.load(str(pre_ct))
    pre_mask_img = nib.load(str(pre_mask))
    post_ct_img = nib.load(str(post_ct))
    post_mask_img = nib.load(str(post_mask))

    pre_ct_vol = pre_ct_img.get_fdata().astype(np.float32)
    pre_mask_vol = pre_mask_img.get_fdata().astype(np.uint8)
    post_ct_vol = post_ct_img.get_fdata().astype(np.float32)
    post_mask_vol = post_mask_img.get_fdata().astype(np.uint8)

    pre_ct_slice = _pick_mid_slice(pre_ct_vol)
    pre_mask_slice = _pick_mid_slice(pre_mask_vol)
    post_ct_slice = _pick_mid_slice(post_ct_vol)
    post_mask_slice = _pick_mid_slice(post_mask_vol)

    _render_overlay(pre_ct_slice, pre_mask_slice, out_dir / f"{args.case_id}_pre_overlay.png", f"{args.case_id} pre")
    _render_overlay(post_ct_slice, post_mask_slice, out_dir / f"{args.case_id}_post_overlay.png", f"{args.case_id} post")

    best_plan_path = Path(args.best_plan_json)
    if best_plan_path.exists():
        _render_text_png(best_plan_path.read_text(encoding="utf-8"), out_dir / "best_plan_json.png", "best_plan.json")

    print(f"[OK] Screenshots saved to: {out_dir}")


if __name__ == "__main__":
    main()
