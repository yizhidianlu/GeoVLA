#!/usr/bin/env bash
# GeoVLA conda env bootstrap — designed for AutoDL A800 80GB.
# Idempotent. Run from repo root.

set -euo pipefail
export GEOVLA_ENV_PREFIX="${GEOVLA_ENV_PREFIX:-/root/autodl-tmp/envs/geovla}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"

CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda not found at $CONDA_SH" >&2; exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"

echo "[env_setup] Target prefix: $GEOVLA_ENV_PREFIX"
echo "[env_setup] HF_HOME: $HF_HOME"

if [[ ! -d "$GEOVLA_ENV_PREFIX" ]]; then
  conda create -p "$GEOVLA_ENV_PREFIX" python=3.10 -y
fi
conda activate "$GEOVLA_ENV_PREFIX"

echo "[env_setup] Python: $(python --version)"
echo "[env_setup] Prefix: $CONDA_PREFIX"

# ----- core ML stack (matches pssa-vla pinning for transferability) -----
python -m pip install --upgrade pip wheel setuptools

# PyTorch 2.4.1 + CUDA 12.4 (A800 driver 580+ supports CUDA 13 runtime, but
# torch 2.4 wheels target CU12.4; runtime is forward-compatible)
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.4.1" "torchvision==0.19.1"

# HF + training infra
python -m pip install \
    "transformers==4.45.2" \
    "accelerate>=0.34" \
    "datasets>=2.21" \
    "huggingface-hub>=0.24" \
    "einops>=0.8" \
    "pyyaml>=6.0" \
    "tqdm>=4.66" \
    "rich>=13" \
    "tensorboard>=2.16" \
    "wandb>=0.18" \
    "opencv-python-headless>=4.10" \
    "pillow>=10.4" \
    "robosuite>=1.4" \
    "mujoco>=3.1"

# Install GeoVLA package itself (editable)
python -m pip install -e .

# Reuse already-installed LIBERO (avoid redownload)
if [[ -d /root/autodl-tmp/LIBERO ]]; then
  python -m pip install -e /root/autodl-tmp/LIBERO || true
fi

echo "[env_setup] Done. To activate later:"
echo "  source $CONDA_SH && conda activate $GEOVLA_ENV_PREFIX"
