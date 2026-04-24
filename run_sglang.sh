#!/bin/bash
# Run sglang, also needs huggingface proxy, and no http proxy.
# Usage:
#   bash run_sglang.sh              # unified mode (prefill + decode together)
#   bash run_sglang.sh prefill      # PD disaggregation: prefill server (GPUs 0-3, TP=4, port 30000)
#   bash run_sglang.sh decode       # PD disaggregation: decode server  (GPUs 4-5, TP=2, port 30001)
#   bash run_sglang.sh router       # PD disaggregation: router         (port 8000)

unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRITON_CACHE_DIR=/tmp/triton_cache

MODE="${1:-unified}"
MODEL="Qwen/Qwen3-30B-A3B"
TRANSFER_BACKEND="${DISAGG_BACKEND:-mooncake}"  # options: mooncake, nixl, mori
UNIFIED_TP="${UNIFIED_TP:-4}"
PREFILL_TP="${PREFILL_TP:-4}"
DECODE_TP="${DECODE_TP:-2}"
PREFILL_CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DECODE_CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-4,5}"
ENABLE_STAGING_BUFFER="${ENABLE_STAGING_BUFFER:-1}"
STAGING_BUFFER_SIZE_MB="${STAGING_BUFFER_SIZE_MB:-64}"
STAGING_POOL_SIZE_MB="${STAGING_POOL_SIZE_MB:-4096}"
ENABLE_HICACHE="${ENABLE_HICACHE:-1}"
ENABLE_DECODE_KVCACHE_OFFLOAD="${ENABLE_DECODE_KVCACHE_OFFLOAD:-1}"
HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-64}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-file}"
HICACHE_PREFETCH_POLICY="${HICACHE_PREFETCH_POLICY:-timeout}"

BASE_ARGS=(
    --model-path "$MODEL"
    --reasoning-parser qwen3
    --disable-custom-all-reduce
    --trust-remote-code
)

# Mooncake RDMA registration can fail when PyTorch uses expandable segments.
if [[ "$TRANSFER_BACKEND" == "mooncake" && ( "$MODE" == "prefill" || "$MODE" == "decode" ) ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# For non-MLA models like Qwen, enable the Mooncake GPU staging buffer when
# prefill and decode use different TP sizes.
if [[ "$TRANSFER_BACKEND" == "mooncake" && "$PREFILL_TP" != "$DECODE_TP" && "$ENABLE_STAGING_BUFFER" == "1" ]]; then
    export SGLANG_DISAGG_STAGING_BUFFER=1
    export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB="$STAGING_BUFFER_SIZE_MB"
    export SGLANG_DISAGG_STAGING_POOL_SIZE_MB="$STAGING_POOL_SIZE_MB"
fi

PREFILL_HICACHE_ARGS=()
DECODE_HICACHE_ARGS=()

if [[ "$ENABLE_HICACHE" == "1" && ( "$MODE" == "prefill" || "$MODE" == "decode" ) ]]; then
    # Use the built-in file backend for same-node PD by default.
    if [[ "$HICACHE_STORAGE_BACKEND" == "file" ]]; then
        export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-/tmp/hicache}"
    fi

    HICACHE_COMMON_ARGS=(
        --page-size "$HICACHE_PAGE_SIZE"
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-storage-backend "$HICACHE_STORAGE_BACKEND"
        --hicache-storage-prefetch-policy "$HICACHE_PREFETCH_POLICY"
    )

    if [[ -n "${HICACHE_SIZE:-}" ]]; then
        HICACHE_COMMON_ARGS+=(--hicache-size "$HICACHE_SIZE")
    fi

    if [[ -n "${HICACHE_STORAGE_BACKEND_EXTRA_CONFIG:-}" ]]; then
        HICACHE_COMMON_ARGS+=(--hicache-storage-backend-extra-config "$HICACHE_STORAGE_BACKEND_EXTRA_CONFIG")
    fi

    PREFILL_HICACHE_ARGS=(
        --enable-hierarchical-cache
        "${HICACHE_COMMON_ARGS[@]}"
    )

    DECODE_HICACHE_ARGS=("${HICACHE_COMMON_ARGS[@]}")
    if [[ "$ENABLE_DECODE_KVCACHE_OFFLOAD" == "1" ]]; then
        DECODE_HICACHE_ARGS+=(--disaggregation-decode-enable-offload-kvcache)
    fi
fi

case "$MODE" in

unified)
    sglang serve "${BASE_ARGS[@]}" \
        --tp "$UNIFIED_TP"
    ;;

prefill)
    CUDA_VISIBLE_DEVICES="$PREFILL_CUDA_VISIBLE_DEVICES" \
    sglang serve "${BASE_ARGS[@]}" \
        --tp "$PREFILL_TP" \
        --host 127.0.0.1 \
        --port 30000 \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
        --disaggregation-bootstrap-port 63465 \
        --disaggregation-ib-device mlx5_1 \
        "${PREFILL_HICACHE_ARGS[@]}"
    ;;

decode)
    CUDA_VISIBLE_DEVICES="$DECODE_CUDA_VISIBLE_DEVICES" \
    sglang serve "${BASE_ARGS[@]}" \
        --tp "$DECODE_TP" \
        --host 127.0.0.1 \
        --port 30001 \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
        --disaggregation-bootstrap-port 63465 \
        --base-gpu-id 0 \
        --disaggregation-ib-device mlx5_5 \
        "${DECODE_HICACHE_ARGS[@]}"
    ;;

router)
    python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill http://127.0.0.1:30000 \
        --decode http://127.0.0.1:30001 \
        --host 0.0.0.0 \
        --port 8000 \
        --prometheus-port 63467
    ;;

*)
    echo "Usage: $0 [unified|prefill|decode|router]"
    exit 1
    ;;
esac
