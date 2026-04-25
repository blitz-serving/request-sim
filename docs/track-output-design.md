# Track-Output Design: Multi-Turn KV Cache Consistency

**Issue**: [blitz-serving/request-sim#22](https://github.com/blitz-serving/request-sim/issues/22)

## Problem

In `prompt-text-hashed` mode, `inflate()` generates random text for hash_id blocks via `TokenSampler`. For multi-turn traces, turn N+1's blocks include turn N's output — but populated with random text instead of the engine's actual output. This causes prefix cache misses in the backend, making benchmarks under-report caching efficiency.

## 0. Formal Problem Statement

request-sim generates `text ∈ String` for each request. The engine processes this text through two transformations before tokenization:

```
token_ids = tokenize(template(message(text)))
```

Where:
- `message : String → [Message]` — wraps text in JSON message structure (e.g., `[{"role":"user","content":text}]`)
- `template : [Message] → String` — applies chat template, inserting magic tokens (`<|im_start|>`, `<|im_end|>`, role markers)
- `tokenize : String → [TokenId]` — BPE encoding to token IDs

**Constraints on `text`**:
1. **Length**: `|tokenize(text)| == trace.input_length` (matching the trace's expected token count)
2. **Block congruence**: If two requests share hash_id H at block position i, then the tokens at `[i·BLOCK_SIZE, (i+1)·BLOCK_SIZE)` are identical

The congruence rule is implemented via `user_prompts: HashMap<u64, String>` — same hash_id always maps to the same text, producing the same tokens at the same position. All three transformations (`message`, `template`, `tokenize`) are deterministic, so congruence is preserved end-to-end.

**The challenge**: Two algebraic properties make this hard:

1. **Template suffix barrier**: The outer `<|im_end|>\n<|im_start|>assistant\n` in Content mode blocks prefix cache hits at the output boundary (Section 2).
2. **Non-distributivity of `enc` over `app`**: `enc(app(b1, b2)) ≠ app(enc(b1), enc(b2))` — BPE tokenization does not distribute over string concatenation. This means splitting output text into blocks and re-assembling changes the token count (Section 2a).

The solution uses `PromptPayload::Messages` with proper multi-turn role structure. Template tokens (`<|im_start|>`, `<|im_end|>`) serve as **pre-tokenization barriers** that make `enc` distributive at message boundaries, while keeping each message's content tokenized as a single unit.

## 1. Modeling the Backend Inference System's Output

### What the engine returns

Both vLLM and SGLang implement the OpenAI Chat Completions API:

- **Streaming**: SSE chunks with `choices[].delta.content` (text per token). vLLM supports `return_token_ids=true` for raw token IDs.
- **Non-streaming**: `choices[].message.content` (full output text).
- **Usage**: `completion_tokens`, `prompt_tokens`, `prompt_tokens_details.cached_tokens` in final chunk.

### Chat template tokens are NOT in the output

Both engines apply `chat_template` before tokenization and strip special tokens during detokenization (`skip_special_tokens=True`). The output text contains only the model's generated content — no `<|im_start|>`, `<|im_end|>`, or role markers.

Sources: vLLM `v1/engine/detokenizer.py:175-188`, SGLang `srt/managers/detokenizer_manager.py:246-278`.

### Prefix caching operates on raw token IDs

- **vLLM**: Block-based SHA256/xxHash of token blocks (typically 16 tokens/block). `v1/core/kv_cache_utils.py:33-107`.
- **SGLang**: Radix tree indexed by token sequences, page-aligned. `srt/mem_cache/radix_cache.py:152-181`.

Cache hit requires exact same token IDs at the same positions. Text similarity is irrelevant.

## 2. The Template Suffix Barrier (Critical Finding)

### Content mode cannot achieve cross-turn prefix cache hits for the output region

In the current architecture, `inflate()` returns `PromptPayload::Content(text)`. The API wraps this as a single user message. The engine produces:

```
Turn N:  [T_prefix][text_N_tokens][T_suffix] → generates [output_tokens]
Turn N+1: [T_prefix][text_N_tokens][delta_tokens][T_suffix]
```

Where `T_prefix = <|im_start|>user\n` and `T_suffix = <|im_end|>\n<|im_start|>assistant\n`.

**Prefix match analysis**:
- `[T_prefix][text_N_tokens]` matches between Turn N and N+1 ✓
- At position `|T_prefix| + |text_N_tokens|`:
  - Turn N's cache has: `T_suffix[0]` = token 151645 (`<|im_end|>`)
  - Turn N+1's input has: `delta_tokens[0]` (random or injected output text)
  - **151645 ≠ delta_tokens[0]** → prefix match terminates ✗

The template suffix creates an **impassable barrier**. No matter what content fills the delta blocks (random text or actual output), the engine's cached `<|im_end|>` at the boundary never matches the content token.

### 2a. Non-distributivity of `enc` over `app`

BPE tokenization does NOT distribute over string concatenation:

```
enc("helloworld") may yield [helloworld_tok]  (1 token)
enc("hello") ++ enc("world") yields [hello_tok, world_tok]  (2 tokens)
```

BPE merges can cross the concatenation boundary when no pre-tokenization split point exists between the two strings.

**How the block system handles this**: `TokenSampler` frames each block with `<|im_start|>` (151644) and `<|im_end|>` (151645). These are **added tokens** — the pre-tokenizer always splits at their positions. So `enc(framed_b1 ++ framed_b2) = enc(framed_b1) ++ enc(framed_b2)`. Validated by `spawn_request_loop_debug`'s assertion `|enc(inflate_text)| == input_length`.

**Why this kills the inject approach**: If we split output text into BLOCK_SIZE-token chunks (no framing), decode each, and store as block entries, the concatenated chunks lack barriers. BPE can merge across chunk boundaries: `enc(chunk_1 ++ chunk_2) ≠ enc(chunk_1) ++ enc(chunk_2)`. The total token count changes — breaking the length constraint.

**Why Messages is correct**: Template tokens surround each message content, creating barriers at the message level. Within each message, content is tokenized as one unit. The output text goes into a single assistant message — `enc` operates on it in one pass, matching the engine's original tokenization. No splitting, no cross-boundary merge risk.

### Messages mode enables full cross-turn prefix caching

With `PromptPayload::Messages`:

```
Turn 1: [{"role":"user","content":"c1"}]
Turn 2: [{"role":"user","content":"c1"}, {"role":"assistant","content":"output"}, {"role":"user","content":"c2"}]
```

Engine template produces:

```
Turn 1: <|im_start|>user\nc1<|im_end|>\n<|im_start|>assistant\n
Turn 2: <|im_start|>user\nc1<|im_end|>\n<|im_start|>assistant\noutput<|im_end|>\n<|im_start|>user\nc2<|im_end|>\n<|im_start|>assistant\n
```

Turn 2's prefix IS Turn 1's cached sequence (including `<|im_end|>\n<|im_start|>assistant\n` and output tokens). Cache hit extends through the entire output region if `tokenize(output_text) == output_token_ids`.

### Quantitative impact

For `input_length = 2000`, `output_length = 500`:

| Mode | Cache hit tokens | Output cached? |
|------|-----------------|----------------|
| Content | ~2003 | No |
| Messages | ~2508 | Yes |

## 3. Token ID Support

vLLM's `return_token_ids=true` returns raw token IDs per SSE chunk. This eliminates BPE roundtrip concerns entirely:

1. Accumulate token IDs from chunks (no text needed)
2. Split into BLOCK_SIZE chunks (exact, no tokenizer)
3. Decode each chunk for `user_prompts` storage

For the Messages architecture, token IDs are even more valuable: they bypass the `encode(decode(tokens)) ≟ tokens` question entirely.

## 4. BPE Tokenizer Idempotency

### `decode(encode(str)) ≠ str` — confirmed

Causes: NFC normalization, special token matching, streaming UTF-8 boundaries. This is why `TokenSampler` validates every generated block with a roundtrip check.

### `encode(decode(tokens)) ≟ tokens` — not always

- **Non-canonical tokenization**: Model can produce [hel, lo] but `encode("hello")` → [hello]. Canonical tokens from `encode()` always roundtrip.
- **Cross-character boundary tokens**: Byte-level BPE can learn merges crossing UTF-8 boundaries. If such a token falls at a block boundary, `decode()` produces U+FFFD, corrupting the text. Extremely rare with Qwen2.

### Re-blocking correctness

`encode(concat(decode(block_1), decode(block_2))) == block_1 ++ block_2` holds because:
- Pre-tokenization (regex split) is text-determined, not position-determined
- BPE merges within pre-tokenization chunks are deterministic
- Block split points don't affect pre-tokenization boundaries

## 5. PromptPayload Design

```
PromptPayload  →  LLMApi::request_json_body()  →  JSON string (POST body)
```

| Variant | Carries | Used by | API wraps as |
|---------|---------|---------|-------------|
| `Content(String)` | Raw text | Bailian, Mooncake (hashed) | `[{"role":"user","content":text}]` |
| `Messages(Value)` | Structured messages array | OpenAI dataset (plain) | Direct `"messages"` value |
| `Body(Value)` | Complete request body | Amadeus-replay (plain) | Gateway envelope |

**Design principle**: Each variant carries exactly the information its API transformation needs — the Curry-Howard "input type" for `request_json_body`.

**For track-output**: The transition from Content to Messages for multi-turn requests is the key architectural change needed. See Fix Plan below.

## 6. Testing Without a Real Backend

- **ConversationGraph**: Synthetic 3-turn JSONL → verify `parent_index`, `delta_hashes`, `parent_output_blocks`.
- **inject_output_blocks**: Create dataset, inject known output, verify `user_prompts` entries.
- **Output text capture**: Synthetic SSE data → verify `output_text` in metrics.
- **TrackOutputState sync**: Tokio test runtime → verify `complete()` / `wait_for_parent()` ordering.
- **BPE roundtrip**: Known tokenizer → verify block split/concat idempotency.
- **Messages construction**: Verify multi-turn Messages array produces correct token count.

## 7. Configuration: Fail Early, Not Silently

`--track-output` does NOT require `--stream`. Non-streaming returns full output text in `message.content`. Misconfigurations fail at parse time with explanatory messages.
