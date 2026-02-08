#!/usr/bin/env python3
"""
为一个 CT NIfTI 生成 Dummy 的 organ+tumor mask（用于 Mock 闭环跑通）。

输出 mask 值域：
  0: background
  1: organ (简化：强度阈值 + 最大连通域近似，失败则全 1)
  2: tumor (在 organ 内随机/中心球体)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DummyMaskConfig:
    organ_threshold_hu: float = -200.0
    tumor_radius_vox: int = 10
    tumor_offset_vox: tuple[int, int, int] = (0, 0, 0)
    tumor_value: int = 2
    organ_value: int = 1


def _largest_cc(binary: np.ndarray) -> np.ndarray:
    try:
        from skimage.measure import label

        lab = label(binary.astype(np.uint8), connectivity=1)
        if lab.max() == 0:
            return binary
        counts = np.bincount(lab.reshape(-1))
        counts[0] = 0
        keep = counts.argmax()
        return (lab == keep)
    except Exception:
        return binary


def build_dummy_mask(ct: np.ndarray, cfg: DummyMaskConfig) -> np.ndarray:
    # ct: (Z, Y, X)
    organ = ct > cfg.organ_threshold_hu
    organ = _largest_cc(organ)
    if organ.sum() < 1000:
        organ = np.ones_like(organ, dtype=bool)

    z, y, x = ct.shape
    cz, cy, cx = (z // 2, y // 2, x // 2)
    oz, oy, ox = cfg.tumor_offset_vox
    cz, cy, cx = cz + oz, cy + oy, cx + ox

    zz, yy, xx = np.ogrid[:z, :y, :x]
    dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    tumor = dist2 <= (cfg.tumor_radius_vox ** 2)
    tumor = tumor & organ

    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[organ] = cfg.organ_value
    mask[tumor] = cfg.tumor_value
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_nii", type=str, required=True, help="输入 CT NIfTI（.nii/.nii.gz）")
    ap.add_argument("--out_mask", type=str, required=True, help="输出 mask NIfTI（.nii.gz）")
    ap.add_argument("--organ_threshold_hu", type=float, default=-200.0)
    ap.add_argument("--tumor_radius_vox", type=int, default=10)
    ap.add_argument("--tumor_offset_vox", type=int, nargs=3, default=(0, 0, 0), metavar=("DZ", "DY", "DX"))
    args = ap.parse_args()

    import nibabel as nib

    ct_path = Path(args.ct_nii)
    out_path = Path(args.out_mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ct_img = nib.load(str(ct_path))
    ct = ct_img.get_fdata(dtype=np.float32)
    # nibabel gives (X,Y,Z) usually; we keep as loaded for alignment and create mask same shape.
    cfg = DummyMaskConfig(
        organ_threshold_hu=args.organ_threshold_hu,
        tumor_radius_vox=args.tumor_radius_vox,
        tumor_offset_vox=tuple(args.tumor_offset_vox),
    )
    mask = build_dummy_mask(ct, cfg)

    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine=ct_img.affine, header=ct_img.header)
    mask_img.set_data_dtype(np.uint8)
    nib.save(mask_img, str(out_path))
    print(f"[OK] Wrote mask: {out_path}")


if __name__ == "__main__":
    main()

