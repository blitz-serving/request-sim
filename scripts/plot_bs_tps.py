#!/usr/bin/env python3
"""Plot BS vs TPS from request-sim feedback-mode benchmark results.

Usage:
    python3 scripts/plot_bs_tps.py \
        --results 'bs30.jsonl:30' 'bs50.jsonl:50' 'bs70.jsonl:70' 'bs90.jsonl:90' \
        --summaries 'bs30-summary.json' 'bs50-summary.json' 'bs70-summary.json' 'bs90-summary.json' \
        --output bs_tps.png

Each --results argument is a pair of 'jsonl_path:bs_limit'. The script computes
TPS = sum(token_count) / (duration_ms / 1000) from the per-request JSONL and
the summary file. This is needed because request-sim's throughput_tps uses
output_length (always 0 for plaintext datasets).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def compute_tps(jsonl_path: str, summary_path: str) -> tuple[int, float, float]:
    """Return (total_tokens, duration_s, tps)."""
    with open(summary_path) as f:
        summary = json.load(f)
    duration_s = float(summary["duration_ms"]) / 1000.0

    total_tokens = 0
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            total_tokens += int(entry.get("token_count", 0))

    tps = total_tokens / duration_s if duration_s > 0 else 0.0
    return total_tokens, duration_s, tps


def main():
    parser = argparse.ArgumentParser(description="Plot BS vs TPS line chart")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Per-request JSONL files in 'path:bs' format",
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        help="Summary JSON files (same order as --results)",
    )
    parser.add_argument("--output", default="bs_tps.png", help="Output image path")
    parser.add_argument("--title", default="Batch Size vs Throughput (TPS)")
    args = parser.parse_args()

    assert len(args.results) == len(args.summaries), (
        f"Mismatch: {len(args.results)} results vs {len(args.summaries)} summaries"
    )

    bs_values = []
    tps_values = []
    tpot_values = []

    for result_spec, summary_path in zip(args.results, args.summaries):
        jsonl_path, bs_str = result_spec.rsplit(":", 1)
        bs = int(bs_str)

        total_tokens, duration_s, tps = compute_tps(jsonl_path, summary_path)

        with open(summary_path) as f:
            summary = json.load(f)
        tpot = float(summary.get("tpot_mean_ms", 0))

        bs_values.append(bs)
        tps_values.append(tps)
        tpot_values.append(tpot)

        print(
            f"BS={bs:3d}  tokens={total_tokens:>7d}  "
            f"duration={duration_s:>6.1f}s  tps={tps:>7.1f}  tpot={tpot:.1f}ms"
        )

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_tps = "#2563eb"
    ax1.set_xlabel("Batch Size (bs_limit)", fontsize=12)
    ax1.set_ylabel("Throughput (tokens/sec)", color=color_tps, fontsize=12)
    ax1.plot(bs_values, tps_values, "o-", color=color_tps, linewidth=2, markersize=8)
    ax1.tick_params(axis="y", labelcolor=color_tps)
    ax1.set_xticks(bs_values)

    for bs, tps in zip(bs_values, tps_values):
        ax1.annotate(
            f"{tps:.0f}",
            (bs, tps),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            color=color_tps,
        )

    color_tpot = "#dc2626"
    ax2 = ax1.twinx()
    ax2.set_ylabel("TPOT (ms)", color=color_tpot, fontsize=12)
    ax2.plot(
        bs_values, tpot_values, "s--", color=color_tpot, linewidth=1.5, markersize=6
    )
    ax2.tick_params(axis="y", labelcolor=color_tpot)

    ax1.set_title(args.title, fontsize=13, pad=10)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
