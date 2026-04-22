"""Parse bench_serving logs and plot throughput vs. TTFT curves."""

import re
import sys
import os
import json
from pathlib import Path

def parse_log(filepath: str) -> dict | None:
    """Extract key metrics from a bench_serving log file."""
    text = Path(filepath).read_text()

    patterns = {
        "total_throughput": r"Throughput:\s+([\d.]+)\s+requests/s,\s+([\d.]+)\s+(?:input|total)\s+tokens/s,\s+([\d.]+)\s+output\s+tokens/s",
        "ttft_avg": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "ttft_p50": r"Median TTFT \(ms\):\s+([\d.]+)",
        "ttft_p99": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "itl_avg": r"Mean ITL \(ms\):\s+([\d.]+)",
        "itl_p50": r"Median ITL \(ms\):\s+([\d.]+)",
        "itl_p99": r"P99 ITL \(ms\):\s+([\d.]+)",
        "e2e_avg": r"Mean E2EL \(ms\):\s+([\d.]+)",
    }

    result = {"file": os.path.basename(filepath)}

    m = re.search(patterns["total_throughput"], text)
    if not m:
        return None
    result["req_per_sec"] = float(m.group(1))
    result["input_tok_per_sec"] = float(m.group(2))
    result["output_tok_per_sec"] = float(m.group(3))

    for key in ["ttft_avg", "ttft_p50", "ttft_p99", "itl_avg", "itl_p50", "itl_p99", "e2e_avg"]:
        m = re.search(patterns[key], text)
        if m:
            result[key] = float(m.group(1))

    rate_m = re.search(r"rate(\d+(?:\.\d+)?)", filepath)
    if rate_m:
        result["request_rate"] = float(rate_m.group(1))

    return result


def print_summary(results_dir: str):
    logs = sorted(Path(results_dir).glob("*.log"))
    if not logs:
        print(f"No .log files found in {results_dir}")
        return

    rows = []
    for log in logs:
        r = parse_log(str(log))
        if r:
            rows.append(r)

    header = f"{'Name':<40} {'Req/s':>7} {'Out tok/s':>10} {'TTFT avg':>10} {'TTFT P99':>10} {'ITL avg':>9} {'E2E avg':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['file']:<40} "
            f"{r.get('req_per_sec', 0):>7.2f} "
            f"{r.get('output_tok_per_sec', 0):>10.1f} "
            f"{r.get('ttft_avg', 0):>10.1f} "
            f"{r.get('ttft_p99', 0):>10.1f} "
            f"{r.get('itl_avg', 0):>9.1f} "
            f"{r.get('e2e_avg', 0):>10.1f}"
        )

    # Also dump raw JSON for further analysis
    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nRaw data saved to {json_path}")


def plot_sweep(results_dir: str):
    """Plot throughput vs TTFT from sweep results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot. Install with: pip install matplotlib")
        return

    logs = sorted(Path(results_dir).glob("sweep_*.log"))
    rows = [r for log in logs if (r := parse_log(str(log)))]

    if len(rows) < 2:
        print("Need at least 2 sweep data points to plot")
        return

    rows.sort(key=lambda r: r.get("request_rate", 0))

    rates = [r["request_rate"] for r in rows]
    throughputs = [r["output_tok_per_sec"] for r in rows]
    ttfts = [r["ttft_avg"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rates, throughputs, "o-", linewidth=2, markersize=6)
    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("Output Throughput (tok/s)")
    ax1.set_title("Throughput vs Request Rate")
    ax1.grid(True, alpha=0.3)

    ax2.plot(throughputs, ttfts, "s-", linewidth=2, markersize=6, color="tab:orange")
    ax2.set_xlabel("Output Throughput (tok/s)")
    ax2.set_ylabel("Avg TTFT (ms)")
    ax2.set_title("Throughput vs TTFT (Pareto curve)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "sweep_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench_results"
    print_summary(results_dir)
    plot_sweep(results_dir)
