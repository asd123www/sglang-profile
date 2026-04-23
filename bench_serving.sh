#!/bin/bash
# Benchmark sglang serving performance with built-in datasets.
# Usage: bash bench_serving.sh [sharegpt|random-short|random-long|sweep|concurrency|loogle]
#
# Prerequisites: sglang server must be running (see run_sglang.sh)

MODEL="lmsys/gpt-oss-120b-bf16"
BASE_URL="http://127.0.0.1:30000"
PORT=30000
BACKEND="sglang"
OUTPUT_DIR="bench_results"
SGLANG_DIR="/Users/bytedance/Documents/sglang"
HICACHE_DIR="$SGLANG_DIR/benchmark/hicache"

mkdir -p "$OUTPUT_DIR"

run_bench() {
    local name="$1"
    shift
    echo "=========================================="
    echo "Running benchmark: $name"
    echo "=========================================="
    python3 -m sglang.bench_serving \
        --backend "$BACKEND" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        "$@" 2>&1 | tee "$OUTPUT_DIR/${name}.log"
    echo ""
}

run_hicache_bench() {
    local name="$1"
    shift
    echo "=========================================="
    echo "Running benchmark (hicache): $name"
    echo "=========================================="
    python3 "$HICACHE_DIR/bench_serving.py" \
        --backend "$BACKEND" \
        --model "$MODEL" \
        --port "$PORT" \
        "$@" 2>&1 | tee "$OUTPUT_DIR/${name}.log"
    echo ""
}

case "${1:-sharegpt}" in

sharegpt)
    # Short-context baseline using real conversation data.
    # ShareGPT has avg ~680 input tokens, ~260 output tokens.
    run_bench "sharegpt_rate2" \
        --dataset-name sharegpt \
        --num-prompts 500 \
        --request-rate 2

    run_bench "sharegpt_rate4" \
        --dataset-name sharegpt \
        --num-prompts 500 \
        --request-rate 4

    run_bench "sharegpt_rate8" \
        --dataset-name sharegpt \
        --num-prompts 500 \
        --request-rate 8
    ;;

random-short)
    # Controlled short-context benchmark with fixed input/output lengths.
    run_bench "random_in512_out128_rate4" \
        --dataset-name random \
        --random-input-len 512 \
        --random-output-len 128 \
        --num-prompts 300 \
        --request-rate 4

    run_bench "random_in512_out128_rate8" \
        --dataset-name random \
        --random-input-len 512 \
        --random-output-len 128 \
        --num-prompts 300 \
        --request-rate 8
    ;;

random-long)
    # Long-context benchmark to stress prefill.
    run_bench "random_in4096_out256_rate1" \
        --dataset-name random \
        --random-input-len 4096 \
        --random-output-len 256 \
        --num-prompts 200 \
        --request-rate 1

    run_bench "random_in4096_out256_rate2" \
        --dataset-name random \
        --random-input-len 4096 \
        --random-output-len 256 \
        --num-prompts 200 \
        --request-rate 2

    run_bench "random_in16384_out256_rate0.5" \
        --dataset-name random \
        --random-input-len 16384 \
        --random-output-len 256 \
        --num-prompts 100 \
        --request-rate 0.5
    ;;

sweep)
    # Full request-rate sweep for throughput vs. TTFT curves.
    # Uses ShareGPT as the workload, sweeps across rates.
    for rate in 1 2 4 6 8 10 12; do
        run_bench "sweep_sharegpt_rate${rate}" \
            --dataset-name sharegpt \
            --num-prompts 500 \
            --request-rate "$rate"
    done
    ;;

concurrency)
    # Fixed concurrency mode (no Poisson arrival, just max-concurrency cap).
    for conc in 4 8 16 32; do
        run_bench "conc${conc}_sharegpt" \
            --dataset-name sharegpt \
            --num-prompts $((conc * 10)) \
            --max-concurrency "$conc"
    done
    ;;

loogle)
    # Long-context benchmark from Strata paper.
    # LooGLE: avg 21K input tokens, ~16 output tokens, 105 documents, 2410 queries.
    # Prefill-dominated RAG workload.
    LOOGLE_DATA="$HICACHE_DIR/longdep_qa.json"
    if [ ! -f "$LOOGLE_DATA" ]; then
        echo "Downloading LooGLE dataset..."
        cd "$HICACHE_DIR"
        git lfs install
        if [ ! -d "LooGLE" ]; then
            git clone https://huggingface.co/datasets/bigainlco/LooGLE
        fi
        unzip -o LooGLE/data.zip
        cd -
    fi

    # Multiturn mode (preserves conversation structure within each document)
    for rate in 1 2 4; do
        run_hicache_bench "loogle_multiturn_rate${rate}" \
            --dataset-path "$LOOGLE_DATA" \
            --dataset-name loogle \
            --request-rate "$rate" \
            --num-prompts 500 \
            --enable-multiturn \
            --disable-shuffle
    done
    ;;

loogle-shared-prefix)
    # LooGLE with shared prefix caching (tests prefix cache hit rates).
    LOOGLE_DATA="$HICACHE_DIR/longdep_qa.json"
    if [ ! -f "$LOOGLE_DATA" ]; then
        echo "LooGLE data not found. Run 'bash bench_serving.sh loogle' first to download."
        exit 1
    fi

    for rate in 1 2 4; do
        run_hicache_bench "loogle_shared_prefix_rate${rate}" \
            --dataset-path "$LOOGLE_DATA" \
            --dataset-name loogle \
            --request-rate "$rate" \
            --num-prompts 500 \
            --enable-shared-prefix \
            --disable-shuffle
    done
    ;;

*)
    echo "Usage: $0 [sharegpt|random-short|random-long|sweep|concurrency|loogle|loogle-shared-prefix]"
    exit 1
    ;;
esac

echo "=========================================="
echo "All benchmarks complete. Results in $OUTPUT_DIR/"
echo "=========================================="
