#!/usr/bin/env python3
"""
Prefix cache hit rate simulator for Bailian multi-turn traces.

Simulates sequential request dispatch against an LRU prefix cache,
computing the theoretical cache hit rate for both Content mode (baseline)
and Messages mode (--track-output).

Usage:
    python scripts/simulate_cache.py --trace test_2turn.jsonl [--cache-blocks 147045] [--block-size 16]

The cache_blocks parameter should match the vLLM engine's num_gpu_blocks
(read from vLLM logs: "cache_config_info with initialization after num_gpu_blocks is: N").
"""

import argparse
import json
import sys
from collections import OrderedDict


class LRUPrefixCache:
    """Simulates vLLM's block-level prefix cache with LRU eviction."""

    def __init__(self, max_blocks: int, block_size: int = 16):
        self.max_blocks = max_blocks
        self.block_size = block_size
        # hash_id -> cached (acts as LRU via OrderedDict)
        self.cache: OrderedDict[int, bool] = OrderedDict()
        self.total_blocks_queried = 0
        self.total_blocks_hit = 0

    def query_and_insert(self, hash_ids: list[int]) -> tuple[int, int]:
        """
        Simulate prefix cache lookup for a request's hash_ids.
        Returns (hit_count, miss_count).

        Prefix matching: cache hits are contiguous from the start.
        Once a miss occurs, all subsequent blocks are misses
        (prefix cache only matches prefixes, not arbitrary positions).
        """
        hit = 0
        miss = 0
        prefix_broken = False

        for hid in hash_ids:
            self.total_blocks_queried += 1
            if not prefix_broken and hid in self.cache:
                hit += 1
                self.total_blocks_hit += 1
                # Move to end (most recently used)
                self.cache.move_to_end(hid)
            else:
                prefix_broken = True
                miss += 1

        # Insert all blocks (after request completes, all blocks are cached)
        for hid in hash_ids:
            if hid in self.cache:
                self.cache.move_to_end(hid)
            else:
                self.cache[hid] = True
                # Evict if over capacity
                while len(self.cache) > self.max_blocks:
                    self.cache.popitem(last=False)

        return hit, miss

    def insert_output_blocks(self, output_hash_ids: list[int]):
        """Cache output blocks after generation (simulates engine caching decode KV)."""
        for hid in output_hash_ids:
            if hid in self.cache:
                self.cache.move_to_end(hid)
            else:
                self.cache[hid] = True
                while len(self.cache) > self.max_blocks:
                    self.cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        if self.total_blocks_queried == 0:
            return 0.0
        return self.total_blocks_hit / self.total_blocks_queried


def load_trace(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_conversation_graph(items: list[dict]) -> dict[int, int | None]:
    """Returns chat_id -> parent_data_index mapping."""
    chat_to_idx = {item["chat_id"]: i for i, item in enumerate(items)}
    parent_index = {}
    for i, item in enumerate(items):
        pid = item["parent_chat_id"]
        if pid in chat_to_idx and chat_to_idx[pid] != i:
            parent_index[i] = chat_to_idx[pid]
        else:
            parent_index[i] = None
    return parent_index


def simulate_content_mode(items: list[dict], cache_blocks: int, block_size: int):
    """
    Simulate Content mode (baseline, no --track-output).

    In Content mode, all hash_ids are packed into a single user message.
    The engine wraps with template: [T_prefix][hash_blocks][T_suffix].
    Template adds ~8 tokens (not block-aligned), but for block-level simulation
    we model cache at the hash_id block level.

    Prefix cache matches hash_ids sequentially. Shared hash_ids between turns
    produce cache hits. Output blocks (delta hash_ids) are always misses because:
    - The engine's cached sequence has T_suffix after the shared blocks
    - Turn N+1 has delta blocks at that position
    - Token mismatch at the boundary breaks prefix matching

    So in Content mode, cache hit = number of shared hash_ids (prefix-contiguous).
    """
    cache = LRUPrefixCache(cache_blocks, block_size)
    parent_index = build_conversation_graph(items)

    print("=== Content Mode (baseline) ===")
    print(f"{'Req':>4} {'Turn':>4} {'Blocks':>6} {'Hit':>4} {'Miss':>5} {'HitRate':>8}")

    for i, item in enumerate(items):
        hash_ids = item["hash_ids"]

        # In Content mode, the prefix match stops at the boundary between
        # shared blocks and T_suffix. We model this by only allowing hits
        # on blocks that were part of the PREVIOUS request in the same conversation.
        if parent_index[i] is not None:
            parent = items[parent_index[i]]
            parent_set = set(parent["hash_ids"])
            # Prefix-contiguous shared blocks
            shared_prefix_len = 0
            for hid in hash_ids:
                if hid in parent_set:
                    shared_prefix_len += 1
                else:
                    break
        else:
            shared_prefix_len = len(hash_ids)  # first turn: no parent, no match expected

        # Query cache with prefix semantics
        hit, miss = cache.query_and_insert(hash_ids)

        # Also "cache" output blocks (engine generates and caches them)
        # These use synthetic hash_ids (output_block_0, output_block_1, ...)
        # to model that output KV is cached but at different positions
        output_len = item["output_length"]
        num_output_blocks = (output_len + block_size - 1) // block_size
        output_hids = [hash(("output", item["chat_id"], j)) for j in range(num_output_blocks)]
        cache.insert_output_blocks(output_hids)

        print(f"{i:4d} {item['turn']:4d} {len(hash_ids):6d} {hit:4d} {miss:5d} {hit/(hit+miss)*100:7.1f}%")

    print(f"\nOverall prefix cache hit rate: {cache.hit_rate*100:.1f}%")
    return cache.hit_rate


def simulate_messages_mode(items: list[dict], cache_blocks: int, block_size: int):
    """
    Simulate Messages mode (--track-output).

    In Messages mode, multi-turn requests use PromptPayload::Messages:
      [{user, turn1_blocks}, {assistant, actual_output}, {user, turn2_remaining_blocks}]

    The engine template produces:
      [T1_prefix][turn1_blocks][T1_suffix_and_T2_prefix][output_tokens][T2_suffix_and_T3_prefix][remaining_blocks][T3_suffix]

    For prefix cache matching, Turn N+1's template output starts with Turn N's
    full cached sequence (including T_suffix + output_tokens). So the prefix
    match extends through the output region.

    At block level: shared hash_ids + output blocks all hit.
    """
    cache = LRUPrefixCache(cache_blocks, block_size)
    parent_index = build_conversation_graph(items)

    # Track output blocks per chat_id for Messages mode
    output_block_map: dict[int, list[int]] = {}

    print("\n=== Messages Mode (--track-output) ===")
    print(f"{'Req':>4} {'Turn':>4} {'Blocks':>6} {'Hit':>4} {'Miss':>5} {'HitRate':>8}")

    for i, item in enumerate(items):
        hash_ids = item["hash_ids"]
        parent_idx = parent_index[i]

        if parent_idx is not None:
            parent = items[parent_idx]
            parent_chat_id = parent["chat_id"]

            # In Messages mode, the query includes:
            # 1. Parent's hash_ids (user msg 1) — should all cache-hit
            # 2. Parent's output blocks (assistant msg) — should cache-hit
            # 3. Delta hash_ids minus output portion (user msg 2) — new, miss

            parent_hids = parent["hash_ids"]
            parent_output_hids = output_block_map.get(parent_chat_id, [])

            parent_set = set(parent_hids)
            delta_hids = [h for h in hash_ids if h not in parent_set]
            output_block_count = (parent["output_length"] + block_size - 1) // block_size
            remaining_hids = delta_hids[output_block_count:]

            # The effective query sequence for prefix cache:
            # [parent_hids] + [parent_output_hids] + [remaining_hids]
            effective_query = parent_hids + parent_output_hids + remaining_hids
        else:
            effective_query = hash_ids

        hit, miss = cache.query_and_insert(effective_query)

        # Cache output blocks with deterministic IDs
        output_len = item["output_length"]
        num_output_blocks = (output_len + block_size - 1) // block_size
        output_hids = [hash(("output", item["chat_id"], j)) for j in range(num_output_blocks)]
        cache.insert_output_blocks(output_hids)
        output_block_map[item["chat_id"]] = output_hids

        print(f"{i:4d} {item['turn']:4d} {len(effective_query):6d} {hit:4d} {miss:5d} {hit/(hit+miss)*100:7.1f}%")

    print(f"\nOverall prefix cache hit rate: {cache.hit_rate*100:.1f}%")
    return cache.hit_rate


def main():
    parser = argparse.ArgumentParser(description="Prefix cache hit rate simulator")
    parser.add_argument("--trace", required=True, help="Path to Bailian JSONL trace")
    parser.add_argument("--cache-blocks", type=int, default=147045,
                        help="Number of GPU KV cache blocks (from vLLM log: num_gpu_blocks)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Tokens per cache block")
    args = parser.parse_args()

    items = load_trace(args.trace)
    print(f"Loaded {len(items)} trace items")
    print(f"Cache: {args.cache_blocks} blocks x {args.block_size} tokens = "
          f"{args.cache_blocks * args.block_size:,} tokens capacity\n")

    baseline_rate = simulate_content_mode(items, args.cache_blocks, args.block_size)
    tracked_rate = simulate_messages_mode(items, args.cache_blocks, args.block_size)

    print(f"\n{'='*50}")
    print(f"Content mode (baseline):     {baseline_rate*100:.1f}%")
    print(f"Messages mode (track-output): {tracked_rate*100:.1f}%")
    print(f"Improvement:                  +{(tracked_rate - baseline_rate)*100:.1f}pp")


if __name__ == "__main__":
    main()
