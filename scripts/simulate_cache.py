#!/usr/bin/env python3
"""
Prefix cache Oracle for Bailian multi-turn traces.

Simulates vLLM's prefix cache: hash-chain-aware, LRU eviction, block-aligned.
Input: cache size (max_blocks) + trace JSONL.
Output: per-request hit_blocks (aligned to BLOCK_SIZE).

Key: vLLM's block hash depends on the ENTIRE prefix (parent-chain hashing).
Two blocks with the same hash_id at the same position but from different
prefix chains are DIFFERENT cache entries. The Oracle models this by keying
on (parent_chain_hash, block_hash) tuples.

Usage:
    python scripts/simulate_cache.py --trace trace.jsonl [--cache-blocks 147889] [--block-size 16]
"""

import argparse
import json
from collections import OrderedDict


class LRUPrefixCache:
    """Simulates vLLM's prefix cache with parent-chain-aware hashing."""

    def __init__(self, max_blocks: int):
        self.max_blocks = max_blocks
        # key = chain_hash (cumulative hash of all blocks up to and including this one)
        # value = True (presence marker)
        self.cache: OrderedDict[int, bool] = OrderedDict()

    @staticmethod
    def chain_hash(parent: int, block_hash: int) -> int:
        """Compute chain hash: hash(parent_chain_hash, block_hash).
        Mirrors vLLM's parent-dependent block hashing."""
        return hash((parent, block_hash))

    def query_and_insert(self, hash_ids: list[int]) -> int:
        """Prefix-match hash_ids against cache using chain hashing.
        Returns number of contiguous hits from block 0."""
        hit = 0
        parent = 0  # initial chain state

        # Phase 1: prefix match
        chain_keys = []
        for hid in hash_ids:
            ck = self.chain_hash(parent, hid)
            chain_keys.append(ck)
            if ck in self.cache:
                hit += 1
                self.cache.move_to_end(ck)
                parent = ck
            else:
                break

        # Phase 2: insert all blocks (with correct chain hashes)
        parent = 0
        for i, hid in enumerate(hash_ids):
            ck = chain_keys[i] if i < len(chain_keys) else self.chain_hash(parent, hid)
            if ck in self.cache:
                self.cache.move_to_end(ck)
            else:
                self.cache[ck] = True
                while len(self.cache) > self.max_blocks:
                    self.cache.popitem(last=False)
            parent = ck

        return hit


def main():
    parser = argparse.ArgumentParser(description="Prefix cache Oracle")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--cache-blocks", type=int, default=147889)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON (for programmatic comparison)")
    args = parser.parse_args()

    items = []
    with open(args.trace) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    cache = LRUPrefixCache(args.cache_blocks)
    results = []

    if not args.json:
        print(f"Trace: {len(items)} requests, cache: {args.cache_blocks} blocks"
              f" × {args.block_size} tokens")
        print(f"{'Req':>4} {'Turn':>4} {'Blocks':>6} {'Hit':>4}"
              f" {'HitTokens':>9} {'HitRate':>8}")

    for i, item in enumerate(items):
        hash_ids = item["hash_ids"]
        hit = cache.query_and_insert(hash_ids)
        hit_tokens = hit * args.block_size
        results.append({"index": i, "hit_blocks": hit, "hit_tokens": hit_tokens})

        if not args.json:
            turn = item.get("turn", "?")
            total = len(hash_ids)
            rate = hit / total * 100 if total > 0 else 0
            print(f"{i:4d} {turn:>4} {total:6d} {hit:4d}"
                  f" {hit_tokens:9d} {rate:7.1f}%")

    if args.json:
        print(json.dumps(results))
    else:
        total_blocks = sum(len(item["hash_ids"]) for item in items)
        total_hits = sum(r["hit_blocks"] for r in results)
        print(f"\nOverall: {total_hits}/{total_blocks} blocks"
              f" = {total_hits / total_blocks * 100:.1f}%")


if __name__ == "__main__":
    main()
