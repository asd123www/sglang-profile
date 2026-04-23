#!/bin/bash
# Run sglang, also needs huggingface proxy, and no http proxy.
# Usage:
#   bash run_sglang.sh              # unified mode (prefill + decode together)
#   bash run_sglang.sh prefill      # PD disaggregation: prefill server (GPUs 0-3, port 30000)
#   bash run_sglang.sh decode       # PD disaggregation: decode server  (GPUs 4-7, port 30001)
#   bash run_sglang.sh router       # PD disaggregation: router         (port 8000)

unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRITON_CACHE_DIR=/tmp/triton_cache

MODE="${1:-unified}"
MODEL="lmsys/gpt-oss-120b-bf16"
TRANSFER_BACKEND="${DISAGG_BACKEND:-mooncake}"  # options: mooncake, nixl, mori
COMMON_ARGS="--model-path $MODEL --tp 4 --reasoning-parser gpt-oss --disable-custom-all-reduce --trust-remote-code"

# Mooncake RDMA registration can fail when PyTorch uses expandable segments.
if [[ "$TRANSFER_BACKEND" == "mooncake" && ( "$MODE" == "prefill" || "$MODE" == "decode" ) ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

case "$MODE" in

unified)
    sglang serve $COMMON_ARGS
    ;;

prefill)
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    sglang serve $COMMON_ARGS \
        --host 127.0.0.1 \
        --port 30000 \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
        --disaggregation-bootstrap-port 63465 \
        --disaggregation-ib-device mlx5_1
    ;;

decode)
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    sglang serve $COMMON_ARGS \
        --host 127.0.0.1 \
        --port 30001 \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
        --disaggregation-bootstrap-port 63465 \
        --base-gpu-id 0 \
        --disaggregation-ib-device mlx5_5
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
