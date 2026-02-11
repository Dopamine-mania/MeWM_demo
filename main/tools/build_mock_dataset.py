#!/usr/bin/env python3
"""
用现有 DICOM（即便只有术前）构造若干例 Mock 数据（默认 5），生成符合 10 列 TSV 规范的列表文件。

做法：
- 从 dicom_root 下自动挑选 2 个“切片数足够”的 CT series 目录
- DICOM -> NIfTI (pre_ct)
- 复制一份当 post_ct
- 基于 CT 生成 dummy pre/post mask（post mask 的肿瘤球体轻微偏移）
- 生成 lists/train_paired.txt & lists/val_paired.txt

输出目录结构：
  out_root/
    hcc/HCC_MOCK_001/{pre_ct,post_ct,pre_mask,post_mask}.nii.gz
    hcc/HCC_MOCK_002/...
    lists/train_paired.txt
    lists/val_paired.txt
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np


class DummyMaskConfig:
    def __init__(
        self,
        organ_threshold_hu: float = -200.0,
        tumor_radius_vox: int = 10,
        tumor_offset_vox: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.organ_threshold_hu = organ_threshold_hu
        self.tumor_radius_vox = tumor_radius_vox
        self.tumor_offset_vox = tumor_offset_vox


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
    organ = ct > cfg.organ_threshold_hu
    organ = _largest_cc(organ)
    if organ.sum() < 1000:
        organ = np.ones_like(organ, dtype=bool)

    d0, d1, d2 = ct.shape
    c0, c1, c2 = (d0 // 2, d1 // 2, d2 // 2)
    o0, o1, o2 = cfg.tumor_offset_vox
    c0, c1, c2 = c0 + o0, c1 + o1, c2 + o2

    g0, g1, g2 = np.ogrid[:d0, :d1, :d2]
    dist2 = (g0 - c0) ** 2 + (g1 - c1) ** 2 + (g2 - c2) ** 2
    tumor = (dist2 <= (cfg.tumor_radius_vox ** 2)) & organ

    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[organ] = 1
    mask[tumor] = 2
    return mask


def _find_series_dirs(dicom_root: Path, min_files: int = 40) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    for d in dicom_root.rglob("*"):
        if not d.is_dir():
            continue
        s = str(d)
        if "Segmentation" in s or "SEG" in s:
            continue
        dcm_files = list(d.glob("*.dcm"))
        if len(dcm_files) >= min_files:
            candidates.append((len(dcm_files), d))
    candidates.sort(key=lambda x: -x[0])
    return [p for _, p in candidates]


def _dicom_dir_to_nifti(dicom_dir: Path, out_nii: Path) -> None:
    import SimpleITK as sitk

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or []
    if not series_ids:
        raise RuntimeError(f"no series found in {dicom_dir}")
    # pick the first series
    series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files)
    image = reader.Execute()
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_nii), useCompression=True)


def _write_tsv_line(cols: list[str]) -> str:
    assert len(cols) == 10
    return "\t".join(cols) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", type=str, default="main/data/数据", help="DICOM 根目录（会自动选若干 series）")
    ap.add_argument("--out_root", type=str, default="main/work/mock_data", help="Mock 数据输出根目录")
    ap.add_argument("--num_cases", type=int, default=5, help="生成多少组 Mock 样本（默认 5）")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dicom_root = Path(args.dicom_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    series_dirs = _find_series_dirs(dicom_root)
    if len(series_dirs) < 1:
        raise SystemExit(f"[ERROR] 在 {dicom_root} 下找不到 CT series 目录（需要至少 1 个且每个 >=40 张 .dcm）")

    # Reuse series if not enough unique dirs.
    chosen = []
    for i in range(args.num_cases):
        chosen.append(series_dirs[i % len(series_dirs)])
    print("[INFO] chosen series dirs:")
    for d in chosen:
        print("  -", d)

    # local imports after env ready
    import nibabel as nib

    rel_root = out_root  # relative paths in TSV will be relative to out_root
    hcc_dir = out_root / "hcc"
    lists_dir = out_root / "lists"
    hcc_dir.mkdir(parents=True, exist_ok=True)
    lists_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i, sd in enumerate(chosen, start=1):
        case_id = f"HCC_MOCK_{i:03d}"
        case_dir = hcc_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        pre_ct = case_dir / "pre_ct.nii.gz"
        post_ct = case_dir / "post_ct.nii.gz"
        pre_mask = case_dir / "pre_mask.nii.gz"
        post_mask = case_dir / "post_mask.nii.gz"

        _dicom_dir_to_nifti(sd, pre_ct)
        shutil.copyfile(pre_ct, post_ct)

        ct_img = nib.load(str(pre_ct))
        ct = ct_img.get_fdata(dtype=np.float32)

        # Randomized tumor geometry per case to simulate different shapes.
        base_r = random.randint(6, 14)
        pre_cfg = DummyMaskConfig(
            tumor_radius_vox=base_r,
            tumor_offset_vox=(random.randint(-4, 4), random.randint(-4, 4), random.randint(-2, 2)),
        )
        post_cfg = DummyMaskConfig(
            tumor_radius_vox=max(4, base_r + random.randint(-3, 3)),
            tumor_offset_vox=(random.randint(-6, 6), random.randint(-6, 6), random.randint(-3, 3)),
        )
        pre_m = build_dummy_mask(ct, pre_cfg)
        post_m = build_dummy_mask(ct, post_cfg)

        nib.save(nib.Nifti1Image(pre_m.astype(np.uint8), affine=ct_img.affine, header=ct_img.header), str(pre_mask))
        nib.save(nib.Nifti1Image(post_m.astype(np.uint8), affine=ct_img.affine, header=ct_img.header), str(post_mask))

        action_text = random.choice(
            [
                "Epirubicin;Lipiodol",
                "Oxaliplatin;Gelatin Sponge",
                "Cisplatin;PVA",
            ]
        )
        survival_time = str(random.choice([12, 24, 48, 60, 72]))
        event = str(random.choice([0, 1]))

        # 10 列 TSV（相对 out_root 的相对路径）
        cols = [
            str((pre_ct.relative_to(rel_root)).as_posix()),
            str((pre_mask.relative_to(rel_root)).as_posix()),
            "-",  # pre_aux_1
            "-",  # pre_aux_2
            str((post_ct.relative_to(rel_root)).as_posix()),
            str((post_mask.relative_to(rel_root)).as_posix()),
            action_text,
            case_id,
            survival_time,
            event,
        ]
        lines.append(_write_tsv_line(cols))

    (lists_dir / "train_paired.txt").write_text("".join(lines), encoding="utf-8")
    (lists_dir / "val_paired.txt").write_text("".join(lines), encoding="utf-8")

    print(f"[OK] mock dataset ready: {out_root}")
    print("  -", lists_dir / "train_paired.txt")
    print("  -", lists_dir / "val_paired.txt")


if __name__ == "__main__":
    main()
