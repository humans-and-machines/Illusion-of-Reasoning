#!/usr/bin/env bash

# Internal helper sourced by both tools/local_env.sh and the conda activate hook.

if [[ -z "${OPENR1_ROOT:-}" ]]; then
  echo "[local-env] OPENR1_ROOT is not set; cannot bind caches." >&2
  return 1
fi

mkdir -p \
  "${OPENR1_ROOT}/.conda_pkgs" \
  "${OPENR1_ROOT}/.conda_envs" \
  "${OPENR1_ROOT}/.local" \
  "${OPENR1_ROOT}/.pip" \
  "${OPENR1_ROOT}/.pip_cache" \
  "${OPENR1_ROOT}/.tmp" \
  "${OPENR1_ROOT}/.hf_home/hub" \
  "${OPENR1_ROOT}/.hf_cache" \
  "${OPENR1_ROOT}/.xdg_cache" \
  "${OPENR1_ROOT}/.cuda_cache" \
  "${OPENR1_ROOT}/.torch_cache" \
  "${OPENR1_ROOT}/wandb" \
  "${OPENR1_ROOT}/.wandb" \
  "${OPENR1_ROOT}/.wandb_cache" \
  "${OPENR1_ROOT}/datasets_cache"

PIP_CONF="${OPENR1_ROOT}/.pip/pip.conf"
if [[ ! -f "${PIP_CONF}" ]] || ! grep -q "${OPENR1_ROOT}/.pip_cache" "${PIP_CONF}" 2>/dev/null; then
  cat > "${PIP_CONF}" <<EOF
[global]
cache-dir = ${OPENR1_ROOT}/.pip_cache
disable-pip-version-check = true
EOF
fi

export OPENR1_ROOT
export CONDA_PKGS_DIRS="${OPENR1_ROOT}/.conda_pkgs"
export CONDA_ENVS_DIRS="${OPENR1_ROOT}/.conda_envs"
export PIP_CACHE_DIR="${OPENR1_ROOT}/.pip_cache"
export PIP_CONFIG_FILE="${PIP_CONF}"
export TMPDIR="${OPENR1_ROOT}/.tmp"
export HF_HOME="${OPENR1_ROOT}/.hf_home"
export HUGGINGFACE_HUB_CACHE="${OPENR1_ROOT}/.hf_home/hub"
export HF_DATASETS_CACHE="${OPENR1_ROOT}/datasets_cache"
export TRANSFORMERS_CACHE="${OPENR1_ROOT}/.hf_cache"
export XDG_CACHE_HOME="${OPENR1_ROOT}/.xdg_cache"
export CUDA_CACHE_PATH="${OPENR1_ROOT}/.cuda_cache"
export TORCH_HOME="${OPENR1_ROOT}/.torch_cache"
export WANDB_DIR="${OPENR1_ROOT}/wandb"
export WANDB_CACHE_DIR="${OPENR1_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${OPENR1_ROOT}/.wandb"
export WANDB_DATA_DIR="${OPENR1_ROOT}/.wandb"

if [[ -z "${OPENR1_LOCAL_ENV_SILENT:-}" ]]; then
  echo "[local-env] All caches bound to ${OPENR1_ROOT}"
fi

return 0
