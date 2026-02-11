# MeWM 端到端跑通指南（Mock 优先）

目标：不追求医学准确性，仅确保 **Pipeline 全链路可运行**（数据输入 → 预处理 → 训练/推理逻辑转起来 → 输出结果文件）。

本仓库所有新产物默认放在 `main/` 下（避免污染客户原始资料）。

---

## Step 1：环境安装（A40）

**推荐一键脚本（conda）**

```bash
bash main/setup_env.sh
```

**或使用 environment.yaml**

```bash
conda env create -f main/environment.yaml
```

激活环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate mewm-a40
```

---

## Step 2：数据准备

### 2.1 使用现有 DICOM 生成 Mock 数据（建议）

> 直接复用术前 CT 伪造术后 CT + Dummy Mask + 10 列 TSV 列表。

```bash
conda run -n mewm-a40 python main/tools/build_mock_dataset.py \
  --dicom_root main/data/数据 \
  --out_root main/work/mock_data \
  --num_cases 5
```

输出结构示例：

```
main/work/mock_data/
  hcc/HCC_MOCK_001/{pre_ct,post_ct,pre_mask,post_mask}.nii.gz
  lists/train_paired.txt
  lists/val_paired.txt
```

### 2.2 DICOM → NIfTI（必经流程）

```bash
conda run -n mewm-a40 python main/tools/dicom_to_nifti.py \
  --dicom_dir main/data/数据/HCC_006/01-22-2000-NA-CT\ AP\ LIVER\ PROT-32853 \
  --out_dir main/work/nifti \
  --max_series 1
```

---

## Step 3：10 列 TSV 规范（核心）

`lists/train_paired.txt` / `lists/val_paired.txt` 每行 10 列（TSV）：

1. `pre_ct_path`
2. `pre_mask_path`
3. `pre_aux_1`（占位 `-`）
4. `pre_aux_2`（占位 `-`）
5. `post_ct_path`
6. `post_mask_path`
7. `action_text`
8. `pair_id`
9. `survival_time_months`
10. `event_indicator`

示例行：

```
hcc/HCC_001/pre_ct.nii.gz	hcc/HCC_001/pre_mask.nii.gz	-	-	hcc/HCC_001/post_ct.nii.gz	hcc/HCC_001/post_mask.nii.gz	Epirubicin;Lipiodol	HCC_001	48	1
```

更完整的字段说明请参考：`main/数据增补清单.md`。

---

## Step 4：Mock 全链路运行（example.py）

```bash
cd main/MeWM
conda run -n mewm-a40 python example.py \
  --mode mock \
  --policy mock \
  --data_root ../work/mock_data \
  --data_list_file ../work/mock_data/lists/train_paired.txt \
  --out_json ../work/mock_outputs/best_plan.json \
  --device cuda:0
```

输出示例：

```
main/work/mock_outputs/best_plan.json
```

---

## Step 5：数据预处理器/TSV 读取验证

```bash
conda run -n mewm-a40 python main/tools/smoke_test_tsv_loader.py \
  --data_root main/work/mock_data \
  --tsv main/work/mock_data/lists/train_paired.txt
```

---

## Step 6：真实模式（可选）

1) 真实权重：如缺失，默认走随机初始化（仍可跑通，但不具备医学意义）  
2) OpenAI 依赖：如不允许联网或无 Key，使用 `--policy mock` 即可

OpenAI Key 方式：

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

---

## 常见问题（FAQ）

1. **`AddChanneld` 报错（monai 版本差异）**  
   已在代码中做兼容：优先 `AddChanneld`，否则回退 `EnsureChannelFirstd`。

2. **`transformers` 拉起 torch/cu12 依赖冲突**  
   已固定 `transformers<5`，避免强制升级 torch。

3. **CUDA 不可用**  
   运行时加：`--device cpu`；速度慢但可跑通。

4. **路径找不到 / 绝对路径问题**  
   所有脚本已改为相对路径或 config 参数；请确保 `--data_root` 与 TSV 相对路径一致。

5. **OpenAI 无法访问或 Key 缺失**  
   使用 `--policy mock`，避免联网与 API 依赖。

---

## 备注

- 所有运行产物默认写入 `main/work/`，不会影响原始数据。
- 当前客户数据主要是 **DICOM CT（pre）**，缺少 post/mask/生存/治疗文本时，请使用 Mock 流程先跑通。
