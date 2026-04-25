#!/usr/bin/env python3
"""
Explain the gap between the block-level simulator (42.9%) and vLLM's
measured prefix cache hit rate (38.4%).

vLLM operates on the FULL token sequence (including template overhead),
and only caches FULL blocks (partial blocks at sequence end are not reusable).
The block-level simulator ignores both of these.
"""

# === From our test trace ===
# Turn 1: input_length=64, output_length=32, hash_ids=[100,101,102,103]
# Turn 2: input_length=160, output_length=32, hash_ids=[100,101,102,103,200,201,202,203,300,301]

# === From vLLM logs ===
# token_count: 33 (for both turns — includes EOS token)
# num_gpu_blocks: 147045
# KV cache size: 2,352,720 tokens → 16 tokens/block (matches BLOCK_SIZE)

BLOCK_SIZE = 16

# === Template overhead (Qwen3 chat template) ===
# <|im_start|>user\n = 3 tokens (151644, user_id, newline_id)
# <|im_end|>\n<|im_start|>assistant\n = 5 tokens (151645, newline, 151644, assistant_id, newline)
T_PREFIX = 3   # <|im_start|>user\n
T_SUFFIX = 5   # <|im_end|>\n<|im_start|>assistant\n
T_MID = 5      # <|im_end|>\n<|im_start|>user\n (between messages)

print("=" * 60)
print("EXACT TOKEN-LEVEL ANALYSIS")
print("=" * 60)

# ============================================================
# Turn 1 (Content mode, same for both baseline and track-output)
# ============================================================
# Engine receives: template([{user, text_64_tokens}])
# Token sequence: [T_PREFIX][64 hash block tokens][T_SUFFIX]
turn1_input_tokens = T_PREFIX + 64 + T_SUFFIX  # = 72
turn1_output_tokens = 33  # from vLLM log (32 + EOS)
turn1_cached = turn1_input_tokens + turn1_output_tokens  # = 105

print(f"\nTurn 1:")
print(f"  Input tokens:  {T_PREFIX} (prefix) + 64 (blocks) + {T_SUFFIX} (suffix) = {turn1_input_tokens}")
print(f"  Output tokens: {turn1_output_tokens} (32 + EOS)")
print(f"  Total cached:  {turn1_cached}")
print(f"  Full blocks cached: {turn1_cached // BLOCK_SIZE} ({turn1_cached // BLOCK_SIZE * BLOCK_SIZE} tokens)")
print(f"  Partial remainder: {turn1_cached % BLOCK_SIZE} tokens (NOT cached as reusable block)")

# ============================================================
# Turn 2 — BASELINE (Content mode)
# ============================================================
# Engine receives: template([{user, text_160_tokens}])
# Token sequence: [T_PREFIX][160 hash block tokens][T_SUFFIX]
turn2_baseline_tokens = T_PREFIX + 160 + T_SUFFIX  # = 168

# Prefix match: [T_PREFIX][64 shared blocks] = 3 + 64 = 67 tokens
# Then Turn 1 cache has T_SUFFIX[0] = 151645, Turn 2 has delta_blocks[0]
# Mismatch! Prefix match = 67 tokens
baseline_match = T_PREFIX + 64  # = 67
baseline_match_blocks = baseline_match // BLOCK_SIZE  # = 4 full blocks = 64 tokens
baseline_match_aligned = baseline_match_blocks * BLOCK_SIZE

total_tokens_baseline = turn1_input_tokens + turn2_baseline_tokens
baseline_hit_rate = baseline_match_aligned / total_tokens_baseline

print(f"\nTurn 2 — BASELINE (Content):")
print(f"  Input tokens: {turn2_baseline_tokens}")
print(f"  Prefix match: {baseline_match} tokens (shared blocks + template prefix)")
print(f"  Block-aligned match: {baseline_match_blocks} blocks = {baseline_match_aligned} tokens")
print(f"  Total tokens queried (both turns): {total_tokens_baseline}")
print(f"  Hit rate: {baseline_match_aligned}/{total_tokens_baseline} = {baseline_hit_rate*100:.1f}%")

# ============================================================
# Turn 2 — TRACK-OUTPUT (Messages mode)
# ============================================================
# Messages: [{user, 64_tokens}, {assistant, output_text}, {user, remaining_tokens}]
# Engine template:
#   <|im_start|>user\n{64 tokens}<|im_end|>\n<|im_start|>assistant\n
#   {output_text}<|im_end|>\n<|im_start|>user\n
#   {remaining_tokens}<|im_end|>\n<|im_start|>assistant\n

# output_text: captured from Turn 1's response
# When re-tokenized, produces ~32 tokens (output_text excludes EOS)
output_text_tokens = 32  # EOS not included in delta.content

# remaining: delta hash_ids minus output blocks
# delta = 6 hash_ids, output_blocks = ceil(32/16) = 2
# remaining = 4 hash_ids × 16 = 64 tokens
remaining_tokens = 4 * BLOCK_SIZE  # = 64

turn2_tracked_tokens = (T_PREFIX + 64 + T_SUFFIX +
                        output_text_tokens + T_MID +
                        remaining_tokens + T_SUFFIX)
# = 3 + 64 + 5 + 32 + 5 + 64 + 5 = 178

# Prefix match with Turn 1's cached sequence:
# Turn 1 cached: [T_PREFIX][64 blocks][T_SUFFIX][33 output tokens] = 105 tokens
# Turn 2 starts: [T_PREFIX][64 blocks][T_SUFFIX][32 output_text tokens]...
# Match: T_PREFIX (3) + 64 blocks + T_SUFFIX (5) = 72 tokens ✓
# Then: output_text (32 tokens) vs cached output (first 32 of 33) = 32 match ✓
# Then: Turn 1 cache has token 33 (EOS), Turn 2 has <|im_end|> (151645)
# EOS for Qwen3 is <|im_end|> = 151645! So this ALSO matches!
# Actually wait - does vLLM cache the EOS token? The EOS token ends generation,
# but it IS part of the cached KV. And <|im_end|> in the template IS token 151645.
# So the match extends to 72 + 33 = 105 tokens!

# But block alignment: 105 / 16 = 6 full blocks (96 tokens) + 9 partial
tracked_match = 72 + output_text_tokens  # = 104 (conservative: exclude EOS match uncertainty)
tracked_match_blocks = tracked_match // BLOCK_SIZE  # = 6 full blocks = 96 tokens
tracked_match_aligned = tracked_match_blocks * BLOCK_SIZE

total_tokens_tracked = turn1_input_tokens + turn2_tracked_tokens
tracked_hit_rate = tracked_match_aligned / total_tokens_tracked

print(f"\nTurn 2 — TRACK-OUTPUT (Messages):")
print(f"  Input tokens: {turn2_tracked_tokens}")
print(f"  Template breakdown: {T_PREFIX}+64+{T_SUFFIX}+{output_text_tokens}+{T_MID}+{remaining_tokens}+{T_SUFFIX}")
print(f"  Prefix match: {tracked_match} tokens (shared blocks + template + output)")
print(f"  Block-aligned match: {tracked_match_blocks} blocks = {tracked_match_aligned} tokens")
print(f"  Total tokens queried (both turns): {total_tokens_tracked}")
print(f"  Hit rate: {tracked_match_aligned}/{total_tokens_tracked} = {tracked_hit_rate*100:.1f}%")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
print(f"{'':>30} {'Simulator':>10} {'Token-level':>12} {'vLLM actual':>12}")
print(f"{'Baseline (Content)':>30} {'28.6%':>10} {f'{baseline_hit_rate*100:.1f}%':>12} {'26.7%':>12}")
print(f"{'Track-output (Messages)':>30} {'42.9%':>10} {f'{tracked_hit_rate*100:.1f}%':>12} {'38.4%':>12}")

print(f"\n{'='*60}")
print(f"GAP EXPLANATION")
print(f"{'='*60}")
print(f"""
The block-level simulator (42.9%) overcounts because it ignores:

1. TEMPLATE OVERHEAD: vLLM processes template(message(text)), adding
   ~8 tokens per message boundary. These increase the denominator
   (total tokens queried) without being modeled by the simulator.

   Simulator denominator: {4 + 10} hash_id blocks = {(4+10)*16} tokens
   Actual denominator:    {total_tokens_tracked} tokens (including template)

2. BLOCK ALIGNMENT: vLLM only caches FULL 16-token blocks. A prefix
   match of {tracked_match} tokens = {tracked_match_blocks} full blocks ({tracked_match_aligned} tokens).
   The remaining {tracked_match % BLOCK_SIZE} tokens in the partial block are NOT reusable.

Together, these explain the gap:
  Simulator:   42.9% (no template, no alignment)
  Token-level: {tracked_hit_rate*100:.1f}% (with template + alignment)
  vLLM actual: 38.4% (real engine)
""")
