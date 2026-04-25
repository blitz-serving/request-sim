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


def verify_congruence(items: list[dict]):
    """
    Verify C2 (block congruence): every hash_id that appears in multiple requests
    always occupies the same block position(s) within those requests' hash_id lists.

    Also verify that shared hash_ids between parent and child are a contiguous prefix.
    """
    parent_index = build_conversation_graph(items)

    # Map hash_id → set of (request_index, position_in_hash_ids)
    hash_positions: dict[int, list[tuple[int, int]]] = {}
    for i, item in enumerate(items):
        for pos, hid in enumerate(item["hash_ids"]):
            hash_positions.setdefault(hid, []).append((i, pos))

    # Check 1: shared hash_ids between parent and child form a contiguous prefix
    prefix_ok = 0
    prefix_fail = 0
    for i, item in enumerate(items):
        if parent_index[i] is None:
            continue
        parent = items[parent_index[i]]
        parent_hids = parent["hash_ids"]
        child_hids = item["hash_ids"]

        # Find the shared prefix length
        shared_len = 0
        for j in range(min(len(parent_hids), len(child_hids))):
            if parent_hids[j] == child_hids[j]:
                shared_len += 1
            else:
                break

        # Check no parent hash_id appears AFTER the prefix in the child
        parent_set = set(parent_hids)
        stray = [h for h in child_hids[shared_len:] if h in parent_set]
        if stray:
            prefix_fail += 1
            print(f"  FAIL: request {i} has {len(stray)} parent hash_ids after prefix break")
        else:
            prefix_ok += 1

    print(f"\n=== Congruence Check (trace-level necessary condition for C2) ===")
    print(f"Shared-prefix property: {prefix_ok} OK, {prefix_fail} FAIL")

    # Check 2: hash_ids appearing in multiple requests are at the same position
    position_ok = 0
    position_fail = 0
    for hid, positions in hash_positions.items():
        if len(positions) <= 1:
            continue
        pos_values = set(p for _, p in positions)
        if len(pos_values) == 1:
            position_ok += 1
        else:
            position_fail += 1
            reqs = [(req, pos) for req, pos in positions]
            print(f"  FAIL: hash_id {hid} at different positions: {reqs}")

    print(f"Position-congruence: {position_ok} OK, {position_fail} FAIL")
    total_shared = sum(1 for hid, pos in hash_positions.items() if len(pos) > 1)
    print(f"Total shared hash_ids: {total_shared}")
    print(f"  Note: this checks the TRACE structure, not runtime string equality.")
    print(f"  Runtime C2 relies on HashMap single-write semantics (code argument, not formal proof).")

    return prefix_fail == 0 and position_fail == 0


def simulate_token_level(items: list[dict], block_size: int = 16):
    """
    Token-level simulation that accounts for template overhead and block alignment.
    This should match vLLM's actual measured hit rate.
    """
    T_PREFIX = 3   # <|im_start|>user\n
    T_SUFFIX = 5   # <|im_end|>\n<|im_start|>assistant\n
    T_MID = 5      # <|im_end|>\n<|im_start|>user\n

    parent_index = build_conversation_graph(items)
    total_input_tokens = 0
    total_hit_tokens = 0

    print("\n=== Token-Level Simulation (with template overhead + block alignment) ===")
    print(f"{'Req':>4} {'Turn':>4} {'Tokens':>7} {'Match':>6} {'Aligned':>8} {'HitRate':>8}")

    # Track cached token count per request (input + output)
    cached_per_request: dict[int, int] = {}

    for i, item in enumerate(items):
        hash_ids = item["hash_ids"]
        parent_idx = parent_index[i]

        if parent_idx is None:
            # First turn: template([{user, text}])
            input_tokens = T_PREFIX + item["input_length"] + T_SUFFIX
            match_tokens = 0
        else:
            parent = items[parent_idx]
            # Messages mode: template([{user, shared}, {asst, output}, {user, remaining}])
            parent_set = set(parent["hash_ids"])
            delta = [h for h in hash_ids if h not in parent_set]
            output_blocks = (parent["output_length"] + block_size - 1) // block_size
            remaining_blocks = len(delta) - output_blocks
            remaining_tokens = max(remaining_blocks * block_size, 0)

            # We captured output_text from the engine; when re-tokenized ≈ output_length tokens
            output_text_tokens = parent["output_length"]

            input_tokens = (T_PREFIX + parent["input_length"] + T_SUFFIX +
                            output_text_tokens + T_MID +
                            remaining_tokens + T_SUFFIX)

            # Prefix match with parent's cached sequence
            parent_cached = cached_per_request.get(parent_idx, 0)
            # Match = min(parent_cached, our prefix up to end of output)
            our_prefix = T_PREFIX + parent["input_length"] + T_SUFFIX + output_text_tokens
            match_tokens = min(parent_cached, our_prefix)

        # Block-align the match
        aligned = (match_tokens // block_size) * block_size

        # Cache this request's full sequence (input + output)
        output_tokens = item["output_length"] + 1  # +1 for EOS
        cached_per_request[i] = input_tokens + output_tokens

        total_input_tokens += input_tokens
        total_hit_tokens += aligned

        hit_rate = aligned / input_tokens if input_tokens > 0 else 0
        print(f"{i:4d} {item['turn']:4d} {input_tokens:7d} {match_tokens:6d} {aligned:8d} {hit_rate*100:7.1f}%")

    overall = total_hit_tokens / total_input_tokens if total_input_tokens > 0 else 0
    print(f"\nOverall token-level hit rate: {overall*100:.1f}%")
    return overall


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

    congruence_ok = verify_congruence(items)

    baseline_rate = simulate_content_mode(items, args.cache_blocks, args.block_size)
    tracked_rate = simulate_messages_mode(items, args.cache_blocks, args.block_size)
    token_rate = simulate_token_level(items, args.block_size)

    print(f"\n{'='*60}")
    print(f"{'Mode':<30} {'Block-level':>12} {'Token-level':>12}")
    print(f"{'Content (baseline)':<30} {baseline_rate*100:>11.1f}% {'—':>12}")
    print(f"{'Messages (track-output)':<30} {tracked_rate*100:>11.1f}% {token_rate*100:>11.1f}%")
    print(f"{'Congruence C2':<30} {'PASS' if congruence_ok else 'FAIL':>12}")
    print(f"\nThe token-level rate accounts for template overhead (~8 tokens/message)")
    print(f"and block alignment (only full {args.block_size}-token blocks are reusable).")


if __name__ == "__main__":
    main()
