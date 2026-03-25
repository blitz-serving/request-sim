# request-sim

Rust benchmark client for LLM inference endpoints. Supports three request dispatch modes: trace replay, stochastic arrival processes, and control-theory feedback loops. Measures TTFT, TPOT, and throughput against TGI / OpenAI / AIBrix APIs.

## Build & Test

```bash
cargo build --release -j64          # binary: target/release/client
cargo build --release --bin mock_server  # test echo server
```

**Validate without HTTP** (debug mode checks inflate() token counts):
```bash
./client --mode trace-replay --api release-with-debug --dataset bailian --dataset-path trace.jsonl \
  --tokenizer tokenizer.json --tokenizer-config tokenizer_config.json \
  --scale-factor 1.0 --endpoint unused --time-in-secs 30
```

## Request Modes

Three modes controlled by `--mode`:

| Mode | Arrival timing | Concurrency | Dataset cycling | Terminates on |
|------|---------------|-------------|-----------------|---------------|
| `trace-replay` (default) | Replays original trace timestamps, scaled by `--scale-factor` | Unbounded concurrent | No | Dataset exhausted OR `--time-in-secs` |
| `random-process` | Stochastic: `--arrival poisson\|uniform`, rate from `--rate` | Unbounded concurrent | Yes (modular indexing) | `--time-in-secs` ONLY |
| `feedback` | Closed-loop: next request sent when a slot frees | `--target-bs` concurrent (semaphore) | No | Dataset exhausted OR `--time-in-secs` |

### Config constraints per mode

- **trace-replay**: `--scale-factor` required. `--begin-time`/`--end-time` apply.
- **random-process**: `--arrival` and `--rate` required. `--scale-factor`/`--begin-time`/`--end-time` ignored. Cyclic — always needs `--time-in-secs` to terminate.
- **feedback**: `--target-bs` (default 1). BS=1 degenerates to queueing-theory closed-loop. `--scale-factor`/`--begin-time`/`--end-time` ignored.
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
  lib.rs             SpinLock, SpinRwLock, timeout_secs_upon_slo()
  dataset.rs         LLMTrace trait + BailianDataset (block=16) + MooncakeDataset (block=512)
  token_sampler.rs   TokenSampler: N producer threads → crossbeam channels → gen_string()
  requester.rs       RequestContext, ArrivalProcess, spawn_request_loop_*, report_loop(), SummaryStats
  mock_server.rs     Hyper echo server (100-300ms random latency)
  apis/
    mod.rs           LLMApi trait, RequestError, global statics (MODEL_NAME, METRIC_PERCENTILES)
    tgi_api.rs       TGI: extracts metrics from response headers
    openai_api.rs    OpenAI: SSE streaming, per-token latency tracking
    aibrix_api.rs    AIBrix: adds routing-strategy header
```

## Architecture

Pipeline: **Load → Warm → Dispatch → Report**

1. `client.rs` parses CLI args, validates config, loads dataset JSONL, creates `TokenSampler`
2. Optionally pre-generates all prompts into a `PromptCache` (--cache tmpfs|file)
3. Dispatch function selected by `--mode`:
   - `trace-replay`: `spawn_request_loop_with_timestamp()` — replays at trace timestamps (scaled by `--scale-factor`), unbounded concurrent tasks
   - `random-process`: `spawn_request_loop_random_process()` — stochastic inter-arrival sleep (Poisson/uniform), cycles dataset, concurrent tasks
   - `feedback`: `spawn_request_loop_feedback()` — semaphore-gated batch size control (`--target-bs`), iterates dataset once
4. `report_loop()` collects results via flume channel → writes per-request JSONL + percentile summary JSON

### Key design decisions

- **Token producers are OS threads** (not tokio tasks) because `tokenizers` encode/decode is CPU-bound and blocking
- **Cache has two modes**: cold path uses `SpinRwLock` for concurrent lazy population; warm path is lock-free (safe because no writers exist after `warm_cache()`)
- **`SpinRwLock`** is writer-prioritized (waiter bit blocks new readers) — see `lib.rs` for bit layout
- **One request = one tokio task** — dispatched sequentially by timestamp, executed concurrently
