#!/usr/bin/env python3
"""Plot BS vs TPS from request-sim feedback-mode benchmark results.

Usage:
    python3 scripts/plot_bs_tps.py \
        --summaries bs30-summary.json bs50-summary.json bs70-summary.json bs90-summary.json \
        --bs 30 50 70 90 \
        --output bs_tps.png

TPS (tokens per second) is the per-user token generation rate: TPS = 1000 / TPOT_mean.
This is the inverse of TPOT (time per output token in ms), measuring how fast each
user receives tokens — NOT aggregate system throughput.
"""

import argparse
import json

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot BS vs TPS line chart")
    parser.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        help="Summary JSON files from request-sim",
    )
    parser.add_argument(
        "--bs",
        nargs="+",
        type=int,
        required=True,
        help="Batch size values (same order as --summaries)",
    )
    parser.add_argument("--output", default="bs_tps.png", help="Output image path")
    parser.add_argument("--title", default="Batch Size vs Per-User TPS")
    args = parser.parse_args()

    assert len(args.summaries) == len(args.bs), (
        f"Mismatch: {len(args.summaries)} summaries vs {len(args.bs)} bs values"
    )

    bs_values = []
    tps_values = []
    tpot_values = []

    for summary_path, bs in zip(args.summaries, args.bs):
        with open(summary_path) as f:
            summary = json.load(f)
        tpot = float(summary["tpot_mean_ms"])
        tps = 1000.0 / tpot if tpot > 0 else 0.0

        bs_values.append(bs)
        tps_values.append(tps)
        tpot_values.append(tpot)

        rs = summary["requests_success"]
        rt = summary["requests_total"]
        print(f"BS={bs:3d}  tpot={tpot:.1f}ms  tps={tps:.1f}  success={rs}/{rt}")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_tps = "#2563eb"
    ax1.set_xlabel("Batch Size (bs_limit)", fontsize=12)
    ax1.set_ylabel("Per-User TPS (tokens/sec)", color=color_tps, fontsize=12)
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
