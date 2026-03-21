#!/bin/bash
# =============================================================================
# setup_env_leonardo.sh  —  source this at the top of every SLURM script
# Leonardo @ CINECA
# =============================================================================

# ---- Modules ---------------------------------------------------------------
module purge
module load python/3.11.7
# Note: no cuda module — PyTorch cu126 bundles its own CUDA runtime.
# Loading a cuda module can cause NVML driver/library version mismatch
# if the node's GPU driver is older than the loaded toolkit.

# ---- Python environment ----------------------------------------------------
source /leonardo_scratch/fast/IscrB_Decentro/mciccone/envs/boom/bin/activate

# ---- Working directory -----------------------------------------------------
export BOOMTITAN_ROOT="${BOOMTITAN_ROOT:-/leonardo/home/userexternal/mciccone/exp/boomtitan}"
cd "${BOOMTITAN_ROOT}"

# ---- Paths -----------------------------------------------------------------
export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/huggingface}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/checkpoints}"
export LOG_PATH="${LOG_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs}"

# ---- W&B -------------------------------------------------------------------
export WANDB_PROJECT="${WANDB_PROJECT:-boomtitan}"
export WANDB_ENTITY="${WANDB_ENTITY:-mciccone}"

# ---- Internet access via reverse SOCKS5 proxy (optional) -------------------
# The proxy is set up by SSH tunneling from your local machine to the login node,
# then forwarded to compute nodes. Port is stored in $HOME/scripts/.leonardo_port.
#
# If the proxy is available, use it for WandB/HF (HTTP/HTTPS only).
# IMPORTANT: Do NOT set ALL_PROXY or SOCKS_PROXY — Gloo and torchft use raw TCP
# sockets that don't speak SOCKS5, and routing them through the tunnel breaks
# inter-replica communication. HTTP_PROXY/HTTPS_PROXY are safe because NCCL/Gloo
# ignore those (only read by HTTP clients like requests/curl).
_PROXY_PORT_FILE="$HOME/scripts/.leonardo_port"
_proxy_up=false
if [[ -f "${_PROXY_PORT_FILE}" ]]; then
    _PROXY_PORT=$(cat "${_PROXY_PORT_FILE}")
    echo "[setup_env] Waiting for proxy on port ${_PROXY_PORT}..."
    for _i in $(seq 1 30); do
        if ss -tlnp 2>/dev/null | grep -q ":${_PROXY_PORT}" || \
           netstat -tlnp 2>/dev/null | grep -q ":${_PROXY_PORT}"; then
            _proxy_up=true; break
        fi
        sleep 2
    done
fi

if [[ "${_proxy_up}" == true ]]; then
    if curl -sf --socks5 "127.0.0.1:${_PROXY_PORT}" --connect-timeout 5 \
            "https://huggingface.co" -o /dev/null 2>/dev/null; then
        export HTTP_PROXY="socks5://127.0.0.1:${_PROXY_PORT}"
        export HTTPS_PROXY="socks5://127.0.0.1:${_PROXY_PORT}"
        export NO_PROXY="localhost,127.0.0.1,.leonardo.local"
        export no_proxy="${NO_PROXY}"
        # Disable gRPC proxy — gRPC (used by torchft) does not reliably respect
        # NO_PROXY and would try to route lighthouse connections through the tunnel.
        export GRPC_PROXY_OVERRIDE=""
        export WANDB_MODE=online
        export HF_DATASETS_OFFLINE=0
        export TRANSFORMERS_OFFLINE=0
        echo "[setup_env] Proxy active on port ${_PROXY_PORT} — WandB/HF online"
    else
        echo "[setup_env] Proxy port listening but SOCKS5 connection failed — running offline"
        export WANDB_MODE=offline
        export HF_DATASETS_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    fi
else
    export WANDB_MODE=offline
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[setup_env] No proxy — running offline"
fi

# ---- SLURM account / partition (used by job scripts for --account/--partition) --
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-IscrB_Decentro}"
export SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"

# ---- NCCL / InfiniBand (Leonardo A100 nodes use Mellanox HDR) --------------
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
# GLOO_SOCKET_IFNAME intentionally not set — let Gloo auto-detect.
export NCCL_IB_HCA=mlx5              # Mellanox HCA on Leonardo A100 nodes
export CUDA_DEVICE_MAX_CONNECTIONS=1024

# ---- Misc ------------------------------------------------------------------
export RUST_LOG="${RUST_LOG:-info}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1
