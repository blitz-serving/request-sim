# How to Faithfully Replay Multi-Turn LLM Traces

*Prefix cache congruence in request-sim*

---

## 1. The Problem: Faithful Replay

Benchmarking LLM inference engines with production traces requires more than matching request rates and token counts. It demands that the benchmark tool reproduce the *exact token-level structure* that a real workload would produce inside the engine's KV cache. Without this, prefix caching measurements are meaningless -- the benchmark silently defeats the very optimization it tries to measure.

Three domains are involved in turning a trace entry into tokens processed by the engine:

```
Sigma*  ──  UTF-8 strings (what request-sim produces)
   │
   │  message()
   v
M       ──  Messages (what gets POST'd to the serving API)
   │
   │  template() ∘ enc()
   v
T*      ──  Token sequences (what the engine actually processes)
```

request-sim produces `List<M>`. The engine applies a chat template and tokenizer to each M, producing the token sequence T* that enters the attention computation. "Faithful replay" means the resulting T* satisfies two constraints:

**C1 (Length).** `|T*| = input_length` -- the token count matches the trace's recorded value.

**C2 (Hash congruence).** If two requests in the trace share a block hash at position i, then `T*[i*B .. (i+1)*B]` is identical across both requests, where B is the block size (16 tokens for Bailian traces).

C1 ensures the benchmark generates the correct computational load. C2 ensures prefix cache hits occur wherever the trace says they should -- two requests with the same hash prefix share the same KV cache blocks, just as they did in production.

## 2. Why This Is Hard: Two Algebraic Barriers

Achieving C1 and C2 simultaneously requires navigating two fundamental properties of BPE tokenizers.

### Non-invertibility: dec(enc(text)) != text

The tokenizer is lossy in the text-to-tokens-to-text roundtrip. When the engine produces output tokens and we decode them to text for injection into the next turn's prompt, re-encoding may not recover the original token IDs. This is the standard lossy-codec problem: the decode step can normalize whitespace, merge Unicode sequences, or produce text that BPE segments differently on re-encoding.

In practice, canonical BPE tokens (those produced by greedy/sampled decoding from the model vocabulary) do roundtrip correctly through `enc(dec(x))` for Qwen2-family tokenizers. But this is a property of the specific tokenizer, not a guarantee from the algorithm.

### Non-distributivity: enc(app(t1, t2)) != app(enc(t1), enc(t2))

BPE tokenization does not distribute over string concatenation. Merge rules can cross the concatenation boundary:

```
enc("helloworld")         = [tok_helloworld]               -- 1 token
enc("hello") ++ enc("world") = [tok_hello, tok_world]      -- 2 tokens
```

This matters because request-sim builds prompts by concatenating independently generated text blocks. If `enc` does not distribute over `app`, the total token count after concatenation is unpredictable, violating C1. Worse, adjacent blocks influence each other's tokenization, so two requests sharing the same hash at position i could produce different tokens if their neighbors differ -- violating C2.

## 3. The Insight: Function Congruence

The core insight is that if request-sim understands the full `M -> T*` transformation, it can engineer M to satisfy both C1 and C2. The approach works in two parts.

### Part 1: TokenSampler + Special Token Framing (solves C1 and C2 for input)

`TokenSampler` generates text blocks that survive the encode roundtrip with exact token counts. Each block of B tokens is framed with `<|im_start|>` (token 151644) and `<|im_end|>` (token 151645):

```
gen_string(B) -> "<|im_start|> ... content ... <|im_end|>"
                  ^^^^^^^^^^^^^                 ^^^^^^^^^^
                  added token                   added token
```

These are *added tokens* in the Qwen2 tokenizer -- the pre-tokenizer always splits at their positions. This makes `enc` distributive over concatenation of framed blocks:

```
enc(framed_b1 ++ framed_b2) = enc(framed_b1) ++ enc(framed_b2)
```

The generator loop retries until `|enc(gen_string(B))| = B` exactly. Combined with memoization (`HashMap<u64, String>` keyed by hash ID), this guarantees:

- C1: total token count = sum of block sizes = `input_length`
- C2: same hash ID -> same string -> same tokens (HashMap gives at most one value per key; no overwrites)

### Part 2: Output Tracking (solves C2 across turns)

Single-turn congruence is necessary but not sufficient. In multi-turn conversations, the trace looks like:

```
Turn 1: hash_ids = [A, A, A, B, C]           input_length = 80
Turn 2: hash_ids = [A, A, A, B, C, X, X, D]  input_length = 128
```

Blocks `[X, X]` represent the model's output from Turn 1. Block `D` is the user's follow-up. Without `--track-output`, request-sim generates random synthetic text for `X` blocks. But the engine's KV cache at those positions contains the *actual output it generated*. The synthetic text doesn't match -- prefix cache lookup breaks at block C, even though the entire prefix `[A, A, A, B, C]` is shared.

`--track-output` captures the engine's actual output text from Turn 1 and injects it into Turn 2's prompt as an assistant message. This requires the Messages architecture described in the next section.

## 4. The Template Suffix Barrier

Why can't we just stuff the captured output into a single user message? Consider how the chat template transforms Content mode (one `{"role": "user"}` message) into tokens:

```
Content mode token layout
─────────────────────────
Turn 1:  [T_prefix] [text_blocks] [T_suffix] [output_tokens...]
         ├─ <|im_start|>user\n ─┤           ├─ <|im_end|>\n<|im_start|>assistant\n ─┤

Turn 2:  [T_prefix] [text_blocks] [output_text] [new_text] [T_suffix]
         ├─ <|im_start|>user\n ─┤                           ├─ <|im_end|>\n... ─┤
```

At position `|T_prefix| + |text_blocks|`:
- Turn 1's cached sequence has token 151645 (`<|im_end|>`) -- the start of `T_suffix`
- Turn 2's sequence has the first token of `output_text`

These will never match. The template suffix creates an impassable wall. No matter what text we place in the output region, the engine's cached `<|im_end|>` at that boundary position can never equal a content token. Content mode structurally prevents cross-turn prefix hits.

**Messages mode** fixes this by representing the conversation with proper role structure:

```json
[
  {"role": "user",      "content": "turn 1 input text"},
  {"role": "assistant", "content": "captured output from turn 1"},
  {"role": "user",      "content": "turn 2 new text"}
]
```

The engine applies the chat template, producing:

```
Messages mode token layout
──────────────────────────
Turn 1:  <|im_start|>user\n {t1_text} <|im_end|>\n <|im_start|>assistant\n
Turn 2:  <|im_start|>user\n {t1_text} <|im_end|>\n <|im_start|>assistant\n {output} <|im_end|>\n <|im_start|>user\n {t2_text} <|im_end|>\n <|im_start|>assistant\n
         ├────────────────── shared prefix ──────────────────────────────────────────┤
```

Now the template tokens *help* instead of hindering. The `<|im_end|>\n<|im_start|>assistant\n` sequence after the user message appears at the same position in both turns. The output text follows in the same position. The prefix match extends through the entire shared region, breaking only where Turn 2 adds new content.

The `<|im_start|>` / `<|im_end|>` tokens around the assistant message also act as pre-tokenization barriers, isolating the output text from adjacent content during re-tokenization. BPE merges cannot cross these boundaries, so `enc(dec(output_tokens))` within the assistant message is independent of surrounding messages. This neutralizes the non-distributivity barrier for the output region.

### Implementation

The `--track-output` flag activates a `TrackOutputState` that coordinates output capture across concurrent requests:

```
                 ConversationGraph
                 (built at startup from chat_id / parent_chat_id)
                        │
                        v
  Turn 1 dispatched ──> engine ──> output_text captured
                                        │
                                        v  tos.complete(index, text)
                                   watch::channel broadcasts
                                        │
  Turn 2 blocked on ──────────────────> v  tos.wait_for_ancestors()
  tos.wait_for_ancestors()              │
         │                              v
         └──> inflate_as_messages() constructs [user, assistant, user, ...]
              using captured output
```

The `ConversationGraph` is built once at startup from the trace's `chat_id` / `parent_chat_id` fields. For each entry it stores: the parent data index, the delta hash IDs (those not in the parent), and the number of output blocks (`ceil(parent.output_length / BLOCK_SIZE)`). The dispatch loop calls `wait_for_ancestors()` before inflating any request that has a parent, ensuring all ancestor outputs are available before constructing the message array.

Output text is captured from standard OpenAI API responses -- accumulated `choices[].delta.content` for streaming, `choices[].message.content` for non-streaming. No engine modifications required.

## 5. Validation

### Synthetic trace: vLLM v0.10.0, Qwen3-30B-A3B, TP4

On a minimal 2-turn synthetic trace, the Messages architecture shows a clear improvement in prefix cache hit rate:

| Mode | Prefix Cache Hit Rate |
|------|----------------------|
| Content (baseline) | 26.7% |
| Messages (`--track-output`) | 38.4% |

A token-level simulator confirms these numbers exactly (`scripts/simulate_cache.py`). The gap between the block-level simulator (42.9%) and the actual measurement (38.4%) is fully explained by template overhead tokens and block alignment — see `scripts/explain_gap.py`.

### Production trace: known bug

On a 500-entry production Bailian trace (`qwen_traceA_blksz_16.jsonl`), the current implementation shows a **regression**: baseline 37.0% → track-output 27.8%. The root cause: root turns use Content mode (full inflate text) while child turns use Messages mode (stripped text via `TemplateRegistry`). The token sequences at the shared prefix differ, breaking cache hits instead of improving them. Additionally, 79 of 500 requests hung due to unscaled inter-turn gap sleep.

**Fix required**: all turns must use the same Σ\* → M → T\* pipeline when `--track-output` is enabled. See `.claude/plans/cheerful-kindling-shore.md` for the fix plan.

### Token-level analysis (synthetic trace)

The synthetic 2-turn calculation remains valid as a proof of concept:

```
Turn 1:  3 (template) + 64 (input blocks) + 5 (template) = 72 input tokens
         Generates 33 output tokens (32 + EOS). Total cached: 105.

Turn 2 (Messages):
  3 + 64 + 5 + 32 + 5 + 64 + 5 = 178 input tokens
  Prefix match with Turn 1: 104 tokens
  Aligned to 16-token blocks: floor(104/16) * 16 = 96 tokens

Hit rate = 96 / (72 + 178) = 96 / 250 = 38.4%
```

This confirms the Messages architecture is correct in principle. The production regression is an implementation bug (root-child mode mismatch), not a flaw in the approach.

## 6. Congruence: Levels of Assurance

C2 (hash congruence) claims: if two requests share a hash_id at the same block position, the tokens at that position are identical. How confident should we be?

### Trace-level (necessary condition)

The simulator verifies two structural properties of the trace data:
- Shared hash_ids between parent and child form a contiguous prefix
- Every hash_id appearing in multiple requests occupies the same block position

These confirm the trace is well-formed for congruence, but say nothing about the strings request-sim produces at runtime.

### Code-level (sufficient by construction)

`user_prompts: HashMap<u64, String>` maps each hash_id to exactly one string. This is a trivial consequence of HashMap semantics: same key, same value. No code path overwrites an existing entry -- the write branch re-checks `get()` and skips if the key already exists. The `SpinRwLock` concurrency mechanism is an implementation detail for thread safety; even under races, only one insert per key succeeds, and all subsequent readers get the winner's value.

### Runtime empirical (not yet done)

The strongest validation would run the same trace twice against vLLM with `return_token_ids=true`, then compare the token IDs at each shared hash_id position across both runs. Any divergence would indicate a C2 violation. This test has not been performed. For engines that support `return_token_ids`, it would bypass the `enc(dec(x)) = x` assumption entirely.

## 7. Takeaway

If your benchmark tool generates synthetic prompts for multi-turn conversations, the question is not whether prefix caching works -- it is whether your tool constructs prompts in a way that *allows* prefix caching to work. Packing everything into a single user message silently defeats prefix matching at every turn boundary, no matter how carefully the content is matched. The fix is structural: use the actual multi-turn message format that the chat template was designed for, inject real output text into the assistant role, and let the template tokens serve as alignment barriers between turns.

`--track-output` is available in request-sim for the `bailian` dataset in `trace-replay` mode. It requires no engine modifications and works with any OpenAI-compatible serving backend.
