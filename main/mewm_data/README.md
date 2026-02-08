# `mewm_data`：10 列 TSV 数据读取/预处理模块

目的：把我们约定的 **10 列 TSV**（pre/post CT + pre/post mask + action + survival）直接喂给训练/推理代码，并在 DataLoader 前完成常见预处理：
- 统一方向到 `RAS`
- 重采样到固定 spacing（默认 1mm）
- 强度归一化到 `[-1, 1]`（默认 HU 裁剪 `[-175, 600]`）
- 基于 `label.pre` 产生多个 `roi_size` patch（默认 `96³`，`num_samples=5`）

## 快速 Smoke Test

```bash
conda run -n mewm-a40 python main/tools/smoke_test_tsv_loader.py \
  --data_root main/work/mock_data \
  --tsv main/work/mock_data/lists/train_paired.txt
```

## 代码用法

```python
from mewm_data.dataset import PairedTSVDataset

ds = PairedTSVDataset(data_root="main/work/mock_data", tsv_path="main/work/mock_data/lists/train_paired.txt")
```

