#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yaml"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda 未找到。请先安装 Miniconda/Anaconda 并确保 conda 在 PATH 中。" >&2
  exit 1
fi

ENV_NAME="$(awk -F': *' 'tolower($1)=="name"{print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "[ERROR] 无法从 ${ENV_FILE} 解析环境名（name: ...）。" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[INFO] 目标环境: ${ENV_NAME}"
echo "[INFO] 环境文件: ${ENV_FILE}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] conda 环境已存在，跳过 create。"
else
  echo "[INFO] 创建 conda 环境（使用 --override-channels，避免受到全局 .condarc 额外 channel 影响）..."
  conda create -n "${ENV_NAME}" -y --override-channels -c conda-forge python=3.10 pip
fi

echo "[INFO] 安装 conda 依赖（pytorch/cuda + 常用科学计算）..."
conda install -n "${ENV_NAME}" -y --override-channels -c pytorch -c nvidia -c conda-forge \
  "pytorch=2.3.*" torchvision torchaudio pytorch-cuda=12.1 \
  numpy scipy pandas pyyaml pillow matplotlib

echo "[INFO] 安装 conda 侧生态依赖（避免 pip 拉起 torch/cu12 大包）..."
conda install -n "${ENV_NAME}" -y --override-channels -c conda-forge \
  monai nibabel simpleitk pydicom dicom2nifti \
  opencv scikit-image tqdm h5py hydra-core omegaconf

echo "[INFO] 安装 pip 依赖（LLM/扩散库等；不应触发 torch 安装）..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  "accelerate>=0.30.0" \
  "diffusers>=0.30.0" \
  "transformers>=4.41.0" \
  "safetensors>=0.4.0" \
  "sentencepiece>=0.2.0" \
  "elasticdeform>=0.5.0" \
  "openai>=1.30.0"

echo "[INFO] 安装完成。激活命令："
echo "  source \"${CONDA_BASE}/etc/profile.d/conda.sh\" && conda activate ${ENV_NAME}"

echo "[INFO] Smoke test..."
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
import torch
import monai
import SimpleITK as sitk
import pydicom
import nibabel as nib
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("monai", monai.__version__)
print("SimpleITK", sitk.Version_VersionString())
print("pydicom", pydicom.__version__)
print("nibabel", nib.__version__)
PY

echo "[INFO] OK"
