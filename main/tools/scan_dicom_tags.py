#!/usr/bin/env python3
"""
Scan DICOM folders and summarize tags for quick inventory.
Outputs:
  - dicom_tag_summary.tsv
  - dicom_tag_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter


POST_KEYWORDS = [
    "post", "after", "follow", "followup", "postt", "post-treatment", "posttreat",
    "tace", "embol", "ablation", "chemo", "therapy", "treated", "postop", "post op"
]

PRE_KEYWORDS = ["pre", "baseline", "pretreat", "pre-treatment", "preop", "pre op"]

SEG_KEYWORDS = ["seg", "segmentation", "rtstruct", "mask", "label"]


def _has_keyword(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def _safe_get(ds, key: str) -> str:
    return str(getattr(ds, key, "") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", type=str, default="main/data/数据")
    ap.add_argument("--out_dir", type=str, default="main/work/reports")
    args = ap.parse_args()

    dicom_root = Path(args.dicom_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect series directories (contain .dcm files)
    series_dirs = {}
    for dcm_path in dicom_root.rglob("*.dcm"):
        series_dirs.setdefault(dcm_path.parent, dcm_path)

    rows = []
    for series_dir, first_file in sorted(series_dirs.items()):
        try:
            import pydicom
            ds = pydicom.dcmread(str(first_file), stop_before_pixels=True, force=True)
        except Exception as e:
            rows.append({
                "series_dir": str(series_dir.relative_to(dicom_root)),
                "patient_id": series_dir.relative_to(dicom_root).parts[0] if series_dir.relative_to(dicom_root).parts else "",
                "num_files": str(len(list(series_dir.glob("*.dcm")))),
                "modality": "",
                "series_description": f"[READ_ERROR] {e}",
                "protocol_name": "",
                "study_date": "",
                "series_date": "",
                "series_time": "",
                "study_uid": "",
                "series_uid": "",
                "image_type": "",
                "has_pre_keyword": "0",
                "has_post_keyword": "0",
                "has_seg_keyword": "0",
                "is_seg": "0",
            })
            continue

        series_desc = _safe_get(ds, "SeriesDescription")
        protocol = _safe_get(ds, "ProtocolName")
        modality = _safe_get(ds, "Modality")
        study_date = _safe_get(ds, "StudyDate")
        series_date = _safe_get(ds, "SeriesDate")
        series_time = _safe_get(ds, "SeriesTime")
        study_uid = _safe_get(ds, "StudyInstanceUID")
        series_uid = _safe_get(ds, "SeriesInstanceUID")
        image_type = _safe_get(ds, "ImageType")

        folder_str = str(series_dir.relative_to(dicom_root))
        text_blob = " ".join([folder_str, series_desc, protocol, modality, image_type])

        has_pre = _has_keyword(text_blob, PRE_KEYWORDS)
        has_post = _has_keyword(text_blob, POST_KEYWORDS)
        has_seg = _has_keyword(text_blob, SEG_KEYWORDS)
        is_seg = modality.upper() == "SEG" or "SEG" in series_desc.upper() or "SEG" in folder_str.upper()

        rows.append({
            "series_dir": folder_str,
            "patient_id": series_dir.relative_to(dicom_root).parts[0] if series_dir.relative_to(dicom_root).parts else "",
            "num_files": str(len(list(series_dir.glob("*.dcm")))),
            "modality": modality,
            "series_description": series_desc,
            "protocol_name": protocol,
            "study_date": study_date,
            "series_date": series_date,
            "series_time": series_time,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "image_type": image_type,
            "has_pre_keyword": "1" if has_pre else "0",
            "has_post_keyword": "1" if has_post else "0",
            "has_seg_keyword": "1" if has_seg else "0",
            "is_seg": "1" if is_seg else "0",
        })

    # write TSV
    tsv_path = out_dir / "dicom_tag_summary.tsv"
    header = [
        "series_dir",
        "patient_id",
        "num_files",
        "modality",
        "series_description",
        "protocol_name",
        "study_date",
        "series_date",
        "series_time",
        "study_uid",
        "series_uid",
        "image_type",
        "has_pre_keyword",
        "has_post_keyword",
        "has_seg_keyword",
        "is_seg",
    ]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r.get(k, "") for k in header) + "\n")

    # summary markdown
    total_series = len(rows)
    patient_count = len({r["patient_id"] for r in rows if r["patient_id"]})
    post_count = sum(1 for r in rows if r["has_post_keyword"] == "1")
    pre_count = sum(1 for r in rows if r["has_pre_keyword"] == "1")
    seg_count = sum(1 for r in rows if r["is_seg"] == "1" or r["has_seg_keyword"] == "1")
    mod_count = Counter(r["modality"] for r in rows if r["modality"])

    md_path = out_dir / "dicom_tag_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# DICOM 元数据扫描摘要\n\n")
        f.write(f"- 总 series 数：{total_series}\n")
        f.write(f"- 覆盖患者数：{patient_count}\n")
        f.write(f"- 含 pre 关键字的 series：{pre_count}\n")
        f.write(f"- 含 post 关键字的 series：{post_count}\n")
        f.write(f"- 可能为分割/标注的 series：{seg_count}\n\n")
        f.write("## Modality 统计\n\n")
        for k, v in mod_count.most_common():
            f.write(f"- {k}: {v}\n")
        f.write("\n")
        f.write(f"详细清单见：`{tsv_path}`\n")

    print(f"[OK] wrote {tsv_path}")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()
