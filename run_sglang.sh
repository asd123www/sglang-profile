#!/bin/bash
# Run sglang, also needs huggingface proxy, and no http proxy.
# Usage:
#   bash run_sglang.sh              # unified mode (prefill + decode together)
#   bash run_sglang.sh pd           # PD disaggregation (needs 8 GPUs: 0-3 prefill, 4-7 decode)

unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

MODEL="lmsys/gpt-oss-120b-bf16"
COMMON_ARGS="--model-path $MODEL --tp 4 --reasoning-parser gpt-oss --disable-custom-all-reduce --trust-remote-code"

case "${1:-unified}" in

unified)
    sglang serve $COMMON_ARGS
    ;;

pd)
    # Prefill server on GPUs 0-3, port 30000
    CUDA_VISIBLE_DEVICES=0,1,2,3 sglang serve $COMMON_ARGS \
        --disaggregation-mode prefill \
        --port 30000 &

    # Decode server on GPUs 4-7, port 30001
    CUDA_VISIBLE_DEVICES=4,5,6,7 sglang serve $COMMON_ARGS \
        --disaggregation-mode decode \
        --port 30001 &

    # Wait for both servers to be ready
    echo "Waiting for servers to start..."
    sleep 30

    # Router on port 8000 (this is the endpoint you send requests to)
    python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill http://127.0.0.1:30000 \
        --decode http://127.0.0.1:30001 \
        --host 0.0.0.0 \
        --port 8000

    wait
    ;;

*)
    echo "Usage: $0 [unified|pd]"
    exit 1
    ;;
esac
