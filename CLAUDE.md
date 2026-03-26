# request-sim

Rust benchmark client for LLM inference endpoints. Supports three request dispatch modes: trace replay, stochastic arrival processes, and control-theory feedback loops. Measures TTFT, TPOT, and throughput against TGI / OpenAI / AIBrix APIs.

## Build & Test

```bash
cargo build --release -j64          # binary: target/release/request-sim
cargo build --release --bin mock_server  # test echo server
```

**Validate without HTTP** (debug mode checks inflate() token counts):
```bash
./request-sim --mode trace-replay --api release-with-debug --dataset bailian --dataset-path trace.jsonl \
  --tokenizer tokenizer.json --tokenizer-config tokenizer_config.json \
  --scale-factor 1.0 --endpoint unused --time-in-secs 30
```

## Request Modes

Three modes controlled by `--mode`:

| Mode | Arrival timing | Concurrency | Dataset cycling | Terminates on |
|------|---------------|-------------|-----------------|---------------|
| `trace-replay` (default) | Replays original trace timestamps, scaled by `--scale-factor` | Unbounded concurrent | No | Dataset exhausted OR `--time-in-secs` |
| `random-process` | Stochastic: `--arrival poisson\|uniform`, rate from `--rate` | Unbounded concurrent | Yes (modular indexing) | `--time-in-secs` ONLY |
| `feedback` | Closed-loop: AIMD controller probes concurrency | `--bs-limit` ceiling, controller adjusts `bs_allowed` | No | Dataset exhausted OR `--time-in-secs` |

### Config constraints per mode

- **trace-replay**: `--scale-factor` required. `--begin-time`/`--end-time` apply.
- **random-process**: `--arrival` and `--rate` required. `--scale-factor`/`--begin-time`/`--end-time` ignored. Cyclic — always needs `--time-in-secs` to terminate.
- **feedback**: `--bs-limit` (default 1) sets the concurrency ceiling. An AIMD controller ramps `bs_allowed` from 1 toward `bs-limit`, retreating when constraints (`--tpot-limit`, `--tps-limit`, `--all-tokens-limit`) are violated. `--tpot-limit`/`--tps-limit` require `--stream`. `--scale-factor`/`--begin-time`/`--end-time` ignored.
- **release-with-debug**: forces trace-replay mode only.

## Conventions

- All dataset structs live in `dataset.rs`; to add a new dataset type, implement the `LLMTrace` trait and add a match arm in `client.rs`
- All API backends live in `src/apis/`; to add a new backend, implement the `LLMApi` trait and add a match arm in `client.rs`
- `TokenSampler` currently hardcodes **Qwen2Tokenizer** special tokens — will panic on other tokenizers
- `unsafe` blocks guard `UnsafeCell<HashMap>` access in dataset cache — safety depends on the `is_warm` / `SpinRwLock` protocol; read the invariants in `inflate()` before modifying
- Metrics flow through `BTreeMap<String, String>` — not typed structs. Keys are string constants like `"s_time"`, `"e_time"`, `"x-first-token-time"`

## File Map

```
src/
  client.rs          CLI args + validate_config() + mode dispatch (entry point)
  lib.rs             SpinLock, SpinRwLock, timeout_secs_upon_slo(), init_basetime(), get_timestamp()
  dataset.rs         LLMTrace trait + BailianDataset (block=16) + MooncakeDataset (block=512) + MiniMaxDataset + PlainTextDataset
  token_sampler.rs   TokenSampler: N producer threads → crossbeam channels → gen_string()
  requester.rs       RequestContext, ArrivalProcess, ControllerConfig, FeedbackState, spawn_request_loop_*, report_loop(), SummaryStats
  mock_server.rs     Hyper echo server (100-300ms random latency)
  apis/
    mod.rs           LLMApi trait, RequestError, InFlightState, global statics (MODEL_NAME, METRIC_PERCENTILES, MAX_TOKENS_CAP)
    tgi_api.rs       TGI: extracts metrics from response headers
    openai_api.rs    OpenAI: SSE streaming, per-token latency tracking
    sgl_api.rs       SGLang: OpenAI-compatible SSE + usage/cached_tokens extraction
    aibrix_api.rs    AIBrix: adds routing-strategy header
```

## Architecture

Pipeline: **Load → Warm → Dispatch → Report**

1. `client.rs` parses CLI args, validates config, loads dataset JSONL, creates `TokenSampler`
2. Optionally pre-generates all prompts into a `PromptCache` (--cache tmpfs|file)
3. Dispatch function selected by `--mode`:
   - `trace-replay`: `spawn_request_loop_with_timestamp()` — replays at trace timestamps (scaled by `--scale-factor`), unbounded concurrent tasks
   - `random-process`: `spawn_request_loop_random_process()` — stochastic inter-arrival sleep (Poisson/uniform), cycles dataset, concurrent tasks
   - `feedback`: `spawn_request_loop_feedback()` — AIMD controller dynamically adjusts concurrency (`bs_allowed`) toward `--bs-limit`, respecting `--tpot-limit`, `--tps-limit`, `--all-tokens-limit` constraints
4. `report_loop()` collects results via flume channel → writes per-request JSONL + percentile summary JSON

### Key design decisions

- **Token producers are OS threads** (not tokio tasks) because `tokenizers` encode/decode is CPU-bound and blocking
- **Cache has two modes**: cold path uses `SpinRwLock` for concurrent lazy population; warm path is lock-free (safe because no writers exist after `warm_cache()`)
- **`SpinRwLock`** is writer-prioritized (waiter bit blocks new readers) — see `lib.rs` for bit layout
- **One request = one tokio task** — dispatched sequentially by timestamp, executed concurrently

## Prompt Text Modes

Two compile-time feature flags control how prompts are constructed:

| Feature | Prompt content | Datasets | `inflate()` signature | `output_length` |
|---------|---------------|----------|----------------------|-----------------|
| *(default)* hashed | Synthetic noise (from block hashes via `TokenSampler`) | `bailian`, `mooncake` | `inflate(index, &TokenSampler)` | Always > 0 (from trace) |
| `prompt-text-plain` | Meaningful real text | `minimax`, `plaintext` | `inflate(index)` | >= 0 (0 = let model EOS) |

Build with plain mode: `cargo build --release --features prompt-text-plain`

### Semantic invariants

- **hashed**: content is noise, so `output_length` must always be set from the trace — the model cannot produce a meaningful EOS on synthetic tokens. `min_tokens = max_tokens = output_length`.
- **plain**: content is meaningful, so `output_length = 0` is valid (model decides when to stop via EOS). When `output_length > 0` (e.g. MiniMaxDataset), it is used as exact constraint.

### `--max-tokens` safety cap

`--max-tokens` provides a ceiling on generated tokens when the dataset does not specify `output_length`:

| `output_length` | `--max-tokens` | `min_tokens` | `max_tokens` |
|---|---|---|---|
| > 0 (from trace) | any | `output_length` | `output_length` |
| 0 (e.g. plaintext) | set | omit | `--max-tokens` |
| 0 (e.g. plaintext) | unset | omit | omit (EOS only) |

Stored as `MAX_TOKENS_CAP: OnceLock<Option<u64>>` in `apis/mod.rs`, read by all API backends in `request_json_body()`.
