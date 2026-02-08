#!/usr/bin/env python3
"""
Smoke test: 读取 10 列 TSV -> MONAI transforms（重采样/裁剪）-> DataLoader 输出 shape。

示例：
  conda run -n mewm-a40 python main/tools/smoke_test_tsv_loader.py \
    --data_root main/work/mock_data \
    --tsv main/work/mock_data/lists/train_paired.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--tsv", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    # Make `main/` importable
    main_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(main_dir))

    from mewm_data.dataset import PairedTSVDataset  # noqa: E402

    ds = PairedTSVDataset(data_root=args.data_root, tsv_path=args.tsv, mode="valid")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    batch = next(iter(dl))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)} {v.dtype} {v.device}")
        else:
            print(f"{k}: {type(v)}")


if __name__ == "__main__":
    main()

