#!/usr/bin/env python3
"""Downsample a MiniMax trace JSONL by spreading conversation chains across time bins.

Each conversation chain (identified by the first two messages' content hash) is
randomly assigned to one of N bins, where N is the downsample factor. Timestamps
are shifted so bin k occupies [k*T, (k+1)*T) where T is the original trace
timespan. This preserves per-chain causality while reducing effective RPS by ~N.

Note: chains sharing the same system prompt + first user message are grouped
together (conservative: preserves causality, may cause slightly uneven load).

Usage
-----
    python downsample_trace.py data/trace.jsonl --factor 36
    python downsample_trace.py data/trace.jsonl --factor 36 --seed 42 -o out.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


def compute_chain_fingerprint(dialogue_input_str: str) -> bytes:
    """Hash the first two messages' text to identify a conversation chain."""
    try:
        di = json.loads(dialogue_input_str)
        msgs = di.get("data", [])
    except (json.JSONDecodeError, TypeError):
        msgs = []

    parts: List[str] = []
    for m in msgs[:2]:
        parts.append(m.get("text", "") if isinstance(m, dict) else "")

    return hashlib.md5("|".join(parts).encode()).digest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downsample MiniMax trace by spreading chains across time bins")
    parser.add_argument("file", help="Input MiniMax trace JSONL file")
    parser.add_argument("--factor", type=int, required=True,
                        help="Downsample factor N (creates N time bins)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible bin assignment")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSONL path (default: <input>-ds<N>.jsonl)")
    args = parser.parse_args()

    if args.factor < 1:
        sys.exit("Error: --factor must be >= 1")

    output_path = args.output
    if output_path is None:
        stem = Path(args.file).stem
        parent = Path(args.file).parent
        output_path = str(parent / f"{stem}-ds{args.factor}.jsonl")

    import random
    rng = random.Random(args.seed)

    # Pass 1: read all lines, extract metadata
    print(f"Reading {args.file} ...", file=sys.stderr)
    lines: List[bytes] = []          # raw line bytes
    timestamps: List[int] = []       # original server_timestamp
    chain_ids: List[bytes] = []      # chain fingerprint

    ts_pattern = re.compile(rb'"server_timestamp"\s*:\s*(\d+)')

    with open(args.file, "rb") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue

            # Extract timestamp via regex (fast)
            ts_match = ts_pattern.search(stripped)
            if not ts_match:
                continue
            ts = int(ts_match.group(1))

            # Parse JSON for chain fingerprint (need dialogue_input)
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            fp = compute_chain_fingerprint(rec.get("dialogue_input", ""))

            lines.append(stripped)
            timestamps.append(ts)
            chain_ids.append(fp)

    n = len(lines)
    if n == 0:
        sys.exit("Error: no valid records found")

    ts_min = min(timestamps)
    ts_max = max(timestamps)
    timespan = ts_max - ts_min
    if timespan == 0:
        timespan = 1  # avoid division by zero for single-request traces

    print(f"  {n} requests, timespan {timespan / 1e9:.3f}s, "
          f"{n / (timespan / 1e9):.1f} RPS", file=sys.stderr)

    # Group by chain fingerprint
    chain_groups: dict[bytes, List[int]] = defaultdict(list)
    for i, fp in enumerate(chain_ids):
        chain_groups[fp].append(i)

    print(f"  {len(chain_groups)} chains", file=sys.stderr)

    # Assign each chain to a random bin
    bin_assignment = [0] * n
    for fp, indices in chain_groups.items():
        k = rng.randint(0, args.factor - 1)
        for i in indices:
            bin_assignment[i] = k

    # Compute new timestamps
    new_timestamps = [
        timestamps[i] + bin_assignment[i] * timespan
        for i in range(n)
    ]

    # Sort by new timestamp
    order = sorted(range(n), key=lambda i: new_timestamps[i])

    # Write output (regex-replace server_timestamp in raw bytes)
    print(f"Writing {output_path} ...", file=sys.stderr)
    ts_replace = re.compile(rb'"server_timestamp"\s*:\s*\d+')

    with open(output_path, "wb") as out:
        for i in order:
            new_line = ts_replace.sub(
                f'"server_timestamp": {new_timestamps[i]}'.encode(),
                lines[i],
                count=1,
            )
            out.write(new_line)
            out.write(b"\n")

    new_min = min(new_timestamps)
    new_max = max(new_timestamps)
    new_span = (new_max - new_min) / 1e9
    print(f"Done. {n} records, new timespan {new_span:.1f}s, "
          f"~{n / new_span:.1f} RPS", file=sys.stderr)


if __name__ == "__main__":
    main()
