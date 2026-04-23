# SGLang Serving Profile

Benchmarking and profiling scripts for serving `gpt-oss-120b-bf16` with [SGLang](https://github.com/sgl-project/sglang) on H800 GPUs.

## Setup

### 1. Install SGLang

```bash
bash install_sglang.sh
```

This installs Rust (needed for tokenizers), clones the sglang repo, and installs it in editable mode.

### 2. Download Model

```bash
bash download_model.sh <YOUR_HF_TOKEN>
```

Replace `<YOUR_HF_TOKEN>` with your Hugging Face access token. Downloads `openai/gpt-oss-120b` and `lmsys/gpt-oss-120b-bf16`.

## Serving

### Unified Mode (default)

Runs prefill and decode together on 4 GPUs with TP=4:

```bash
bash run_sglang.sh
```

The server listens on `http://127.0.0.1:30000`.

### PD Disaggregation Mode

Runs prefill and decode on separate GPU groups (requires 8 GPUs):

```bash
bash run_sglang.sh pd
```

This starts three processes:

| Process | GPUs | Port | Role |
|---------|------|------|------|
| Prefill server | 0-3 | 30000 | Handles prompt prefill |
| Decode server | 4-7 | 30001 | Handles token generation |
| Router | — | 8000 | Routes requests between prefill and decode |

Send requests to the **router** at `http://127.0.0.1:8000`.

### Quick Test

```bash
python3 client.py
```

## Benchmarking

All benchmarks use `bench_serving.sh`. The server must be running first.

```bash
bash bench_serving.sh <mode>
```

### Benchmark Modes

| Mode | Command | Description |
|------|---------|-------------|
| `sharegpt` | `bash bench_serving.sh sharegpt` | Short-context baseline (~680 input tokens). Rates: 2, 4, 8 |
| `random-short` | `bash bench_serving.sh random-short` | Fixed 512 input / 128 output tokens. Rates: 4, 8 |
| `random-long` | `bash bench_serving.sh random-long` | Long context (4K, 16K input tokens). Rates: 0.5, 1, 2 |
| `sweep` | `bash bench_serving.sh sweep` | Rate sweep (1-12) for throughput vs. TTFT curves |
| `concurrency` | `bash bench_serving.sh concurrency` | Fixed concurrency caps (4, 8, 16, 32) |
| `loogle` | `bash bench_serving.sh loogle` | LooGLE long-context benchmark (~21K input tokens). Rates: 1, 2, 4 |
| `loogle-shared-prefix` | `bash bench_serving.sh loogle-shared-prefix` | LooGLE with shared prefix caching |

### Metrics

Each benchmark reports:

- **TTFT** — Time to first token (prefill latency)
- **TPOT** — Time per output token (decode latency per token)
- **ITL** — Inter-token latency
- **E2E** — End-to-end request latency
- **Throughput** — Requests/s, input tok/s, output tok/s

### Viewing Results

Logs are saved to `bench_results/`. To print a summary table and generate plots:

```bash
python3 plot_results.py bench_results/
```

### Workload Characteristics

| Dataset | Avg Input | Avg Output | Type |
|---------|-----------|------------|------|
| ShareGPT | ~680 | ~260 | Short-context conversations |
| Random (short) | 512 | 128 | Controlled synthetic |
| Random (long) | 4K-16K | 256 | Stress prefill |
| LooGLE | ~21K | ~16 | Long-context RAG (prefill-dominated) |

## File Overview

```
├── install_sglang.sh       # Install sglang from source
├── download_model.sh        # Download model weights (requires HF token)
├── run_sglang.sh            # Start sglang server (unified or PD mode)
├── client.py                # Quick smoke test
├── bench_serving.sh         # Run benchmarks
├── plot_results.py          # Parse logs, print summary, generate plots
└── bench_results/           # Benchmark output logs
```
