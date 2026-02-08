#!/usr/bin/env python3
"""
DICOM -> NIfTI 转换工具（面向 MeWM 数据准备）。

特点：
- 支持输入目录包含一个或多个 series（基于 GDCM 发现）
- 默认跳过 DICOM-SEG（Modality=SEG）
- 输出 .nii.gz + 一个简单的 metadata.json（可选）

示例：
  python main/tools/dicom_to_nifti.py \
    --dicom_dir main/data/数据/HCC_009/02-15-1998-NA-PP*/103.000000-LIVER* \
    --out_dir main/work/nifti/HCC_009_pre
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def _sanitize_filename(s: str, max_len: int = 120) -> str:
    s = s.strip().replace(os.sep, "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "series"
    return s[:max_len]


def _is_dicom_seg(series_files: list[str]) -> bool:
    try:
        import pydicom

        ds = pydicom.dcmread(series_files[0], stop_before_pixels=True, force=True)
        modality = getattr(ds, "Modality", None)
        return modality == "SEG"
    except Exception:
        return False


def _read_series_sitk(series_files: list[str]):
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files)
    image = reader.Execute()
    return image


def _series_meta(series_files: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        import pydicom

        ds = pydicom.dcmread(series_files[0], stop_before_pixels=True, force=True)
        for k in [
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SeriesNumber",
            "SeriesDescription",
            "Modality",
            "StudyDate",
            "SeriesDate",
        ]:
            v = getattr(ds, k, None)
            if v is None:
                continue
            meta[k] = str(v)
    except Exception as e:
        meta["pydicom_error"] = repr(e)
    meta["num_files"] = len(series_files)
    meta["first_file"] = str(series_files[0])
    return meta


def convert_directory(dicom_dir: Path, out_dir: Path, write_json: bool, max_series: int | None) -> list[Path]:
    import SimpleITK as sitk

    dicom_dir = dicom_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or []
    if not series_ids:
        raise SystemExit(f"[ERROR] 未在目录中发现 DICOM Series：{dicom_dir}")

    written: list[Path] = []
    for idx, series_id in enumerate(series_ids):
        if max_series is not None and idx >= max_series:
            break

        series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
        if not series_files:
            continue
        if _is_dicom_seg(series_files):
            print(f"[SKIP] Modality=SEG series_id={series_id}")
            continue

        meta = _series_meta(series_files)
        base_name = meta.get("SeriesDescription") or meta.get("SeriesNumber") or series_id
        base_name = _sanitize_filename(str(base_name))
        nii_path = out_dir / f"{base_name}_{idx:02d}.nii.gz"

        print(f"[INFO] Reading series {idx+1}/{len(series_ids)}: {base_name}")
        image = _read_series_sitk(series_files)
        sitk.WriteImage(image, str(nii_path), useCompression=True)
        written.append(nii_path)

        if write_json:
            json_path = nii_path.with_suffix("").with_suffix(".json")
            json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_dir", type=str, required=True, help="包含一个或多个 DICOM series 的目录")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录（写入 nii.gz）")
    ap.add_argument("--write_json", action="store_true", help="额外写出每个 series 的 metadata.json")
    ap.add_argument("--max_series", type=int, default=None, help="最多转换多少个 series（用于快速测试）")
    args = ap.parse_args()

    dicom_dir = Path(args.dicom_dir)
    out_dir = Path(args.out_dir)

    written = convert_directory(dicom_dir, out_dir, write_json=args.write_json, max_series=args.max_series)
    if not written:
        raise SystemExit("[ERROR] 未写出任何 NIfTI（可能全部被跳过，或目录不含可读 CT series）")
    print("[OK] Written:")
    for p in written:
        print("  -", p)


if __name__ == "__main__":
    main()

