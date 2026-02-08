from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import Dataset

from .tsv import read_paired_tsv_10col


@dataclass(frozen=True)
class TransformConfig:
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    roi_size: tuple[int, int, int] = (96, 96, 96)
    a_min: float = -175.0
    a_max: float = 600.0
    b_min: float = -1.0
    b_max: float = 1.0
    num_samples: int = 5


def build_transforms(cfg: TransformConfig, mode: str):
    from monai.transforms import (
        Compose,
        CropForegroundd,
        Orientationd,
        RandCropByLabelClassesd,
        ScaleIntensityRanged,
        Spacingd,
        SpatialPadd,
        ToTensord,
        LoadImaged,
    )
    try:
        from monai.transforms import AddChanneld as _AddChanneld

        def AddChanneld(*args, **kwargs):  # type: ignore
            return _AddChanneld(*args, **kwargs)

    except ImportError:
        from monai.transforms import EnsureChannelFirstd as _EnsureChannelFirstd

        def AddChanneld(*args, **kwargs):  # type: ignore
            kwargs.setdefault("channel_dim", "no_channel")
            return _EnsureChannelFirstd(*args, **kwargs)

    keys_img = ["image.pre", "image.post"]
    keys_lbl = ["label.pre", "label.post"]
    keys_all = keys_img + keys_lbl

    xforms = [
        LoadImaged(keys=keys_all),
        AddChanneld(keys=keys_all),
        Orientationd(keys=keys_all, axcodes="RAS"),
        Spacingd(
            keys=keys_all,
            pixdim=cfg.spacing,
            mode=("bilinear", "bilinear", "nearest", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=keys_img,
            a_min=cfg.a_min,
            a_max=cfg.a_max,
            b_min=cfg.b_min,
            b_max=cfg.b_max,
            clip=True,
        ),
        CropForegroundd(keys=keys_all, source_key="image.pre"),
        SpatialPadd(keys=keys_all, spatial_size=cfg.roi_size, mode="constant"),
    ]

    # 训练/验证统一用“按 label.pre 裁剪”的方式，保证能产生多个 patch
    xforms.append(
        RandCropByLabelClassesd(
            keys=keys_all,
            label_key="label.pre",
            spatial_size=cfg.roi_size,
            ratios=[0.1, 0.1, 1.0],
            num_classes=3,
            num_samples=cfg.num_samples,
            image_key="image.pre",
            image_threshold=-1,
        )
    )

    xforms.append(ToTensord(keys=keys_all))
    return Compose(xforms)


class PairedTSVDataset(Dataset):
    """
    读取 10 列 TSV，并在 __getitem__ 返回：
      - input_CT_pre / input_CT_post: (num_samples, 1, D, H, W) 拼接后的 tensor
      - label_CT_pre / label_CT_post: 同上
      - action_text / pair_id / survival_time_months / event_indicator

    注意：RandCropByLabelClassesd 会返回 list[dict]，因此这里会把 list 里的每个 crop 拼接成 batch。
    """

    def __init__(self, data_root: str, tsv_path: str, mode: str = "valid", transform_cfg: TransformConfig | None = None):
        self.data_root = data_root
        self.tsv_path = tsv_path
        self.mode = mode
        self.rows = read_paired_tsv_10col(tsv_path)
        self.transform_cfg = transform_cfg or TransformConfig()
        self.transform = build_transforms(self.transform_cfg, mode=mode)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        item = {
            "image.pre": os.path.join(self.data_root, r.pre_ct),
            "image.post": os.path.join(self.data_root, r.post_ct),
            "label.pre": os.path.join(self.data_root, r.pre_mask),
            "label.post": os.path.join(self.data_root, r.post_mask),
            "action_text": r.action_text,
            "pair_id": r.pair_id,
            "survival_time_months": float(r.survival_time_months),
            "event_indicator": int(r.event_indicator),
        }

        out = self.transform(item)
        # RandCropByLabelClassesd returns list[dict]
        if isinstance(out, list):
            pre = torch.cat([d["image.pre"] for d in out], dim=0)
            post = torch.cat([d["image.post"] for d in out], dim=0)
            pre_lbl = torch.cat([d["label.pre"] for d in out], dim=0)
            post_lbl = torch.cat([d["label.post"] for d in out], dim=0)
        else:
            pre, post, pre_lbl, post_lbl = out["image.pre"], out["image.post"], out["label.pre"], out["label.post"]

        return {
            "input_CT_pre": pre,
            "input_CT_post": post,
            "label_CT_pre": pre_lbl,
            "label_CT_post": post_lbl,
            "action_text": r.action_text,
            "pair_id": r.pair_id,
            "survival_time_months": torch.tensor([float(r.survival_time_months)], dtype=torch.float32),
            "event_indicator": torch.tensor([float(r.event_indicator)], dtype=torch.float32),
        }
