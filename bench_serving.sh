#!/bin/bash
# Benchmark sglang serving performance with built-in datasets.
# Usage: bash bench_serving.sh [sharegpt|random-short|random-long|sweep]
#
# Prerequisites: sglang server must be running (see run_sglang.sh)

MODEL="lmsys/gpt-oss-120b-bf16"
BASE_URL="http://127.0.0.1:30000"
BACKEND="sglang"
OUTPUT_DIR="bench_results"

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

*)
    echo "Usage: $0 [sharegpt|random-short|random-long|sweep|concurrency]"
    exit 1
    ;;
esac

echo "=========================================="
echo "All benchmarks complete. Results in $OUTPUT_DIR/"
echo "=========================================="
