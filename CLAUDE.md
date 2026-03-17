# request-sim

Rust trace-replay benchmark client for LLM inference endpoints. Replays timestamped production traces against TGI / OpenAI / AIBrix APIs, measuring TTFT, TPOT, and throughput.

## Build & Test

```bash
cargo build --release -j64          # binary: target/release/client
cargo build --release --bin mock_server  # test echo server
```

**Validate without HTTP** (debug mode checks inflate() token counts):
```bash
./client --api release-with-debug --dataset bailian --dataset-path trace.jsonl \
  --tokenizer tokenizer.json --tokenizer-config tokenizer_config.json \
  --scale-factor 1.0 --endpoint unused --time-in-secs 30
```

## Conventions

- All dataset structs live in `dataset.rs`; to add a new dataset type, implement the `LLMTrace` trait and add a match arm in `client.rs`
- All API backends live in `src/apis/`; to add a new backend, implement the `LLMApi` trait and add a match arm in `client.rs`
- `TokenSampler` currently hardcodes **Qwen2Tokenizer** special tokens — will panic on other tokenizers
- `unsafe` blocks guard `UnsafeCell<HashMap>` access in dataset cache — safety depends on the `is_warm` / `SpinRwLock` protocol; read the invariants in `inflate()` before modifying
- Metrics flow through `BTreeMap<String, String>` — not typed structs. Keys are string constants like `"s_time"`, `"e_time"`, `"x-first-token-time"`

## File Map

```
src/
  client.rs          CLI args + main orchestration (entry point)
  lib.rs             SpinLock, SpinRwLock, timeout_secs_upon_slo()
  dataset.rs         LLMTrace trait + BailianDataset (block=16) + MooncakeDataset (block=512)
  token_sampler.rs   TokenSampler: N producer threads → crossbeam channels → gen_string()
  requester.rs       spawn_request_loop_with_timestamp(), report_loop(), SummaryStats
  mock_server.rs     Hyper echo server (100-300ms random latency)
  apis/
    mod.rs           LLMApi trait, RequestError, global statics (MODEL_NAME, METRIC_PERCENTILES)
    tgi_api.rs       TGI: extracts metrics from response headers
    openai_api.rs    OpenAI: SSE streaming, per-token latency tracking
    aibrix_api.rs    AIBrix: adds routing-strategy header
```

## Architecture

Pipeline: **Load → Warm → Replay → Report**

1. `client.rs` parses CLI args, loads dataset JSONL, creates `TokenSampler`
2. `warm_cache()` pre-generates all unique token blocks into a HashMap (sets `is_warm=true`)
3. `spawn_request_loop_*()` replays requests at trace timestamps (scaled by `--scale-factor`), calling `inflate()` (lock-free in warm mode) and spawning async HTTP tasks
4. `report_loop()` collects results via flume channel → writes per-request JSONL + percentile summary JSON

### Key design decisions

- **Token producers are OS threads** (not tokio tasks) because `tokenizers` encode/decode is CPU-bound and blocking
- **Cache has two modes**: cold path uses `SpinRwLock` for concurrent lazy population; warm path is lock-free (safe because no writers exist after `warm_cache()`)
- **`SpinRwLock`** is writer-prioritized (waiter bit blocks new readers) — see `lib.rs` for bit layout
- **One request = one tokio task** — dispatched sequentially by timestamp, executed concurrently
