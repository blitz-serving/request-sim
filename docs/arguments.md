# Command-line Arguments

Complete CLI reference for **request-sim**. All arguments use `--kebab-case`.

## Core (always required)

| Argument | Type | Description |
|----------|------|-------------|
| `--endpoint` | `String` | Target HTTP endpoint (e.g. `http://localhost:8080/v1/chat/completions`) |
| `--api`, `-a` | `String` | API backend: `openai`, `sgl`, `tgi`, `aibrix`, `release-with-debug` |
| `--dataset`, `-d` | `String` | Dataset type (see [Datasets](#datasets)) |
| `--dataset-path` | `String` | Path to dataset file |

### Datasets

Available datasets depend on the compile-time feature flag:

| Build | Feature flag | Datasets |
|-------|-------------|----------|
| Default (hashed) | *(none)* | `bailian`, `mooncake` |
| Plain text | `--features prompt-text-plain` | `plaintext`, `openai` |

## Mode Selection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | `String` | `trace-replay` | Dispatch mode: `trace-replay`, `random-process`, `feedback` |

### trace-replay mode

Replays requests at original trace timestamps, scaled by `--scale-factor`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scale-factor` | `f64` | *(required)* | Time scaling. `2.0` = 2x speed (more requests per wall-clock second) |
| `--begin-time` | `u64` | None | Filter: only replay requests with trace timestamp >= this (ms) |
| `--end-time` | `u64` | None | Filter: only replay requests with trace timestamp <= this (ms) |

### random-process mode

Stochastic arrival times. Cycles through dataset indefinitely until `--time-in-secs`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--arrival` | `String` | *(required)* | Arrival process: `poisson` or `uniform` |
| `--rate` | `f64` | *(required)* | Request rate in req/s (must be > 0) |

### feedback mode

AIMD closed-loop controller. Probes concurrency from 1 toward `--bs-limit`, retreating when constraints are violated.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bs-limit` | `usize` | `1` | Max concurrent in-flight requests (concurrency ceiling) |
| `--controller-interval` | `f64` | `0.2` | AIMD tick interval in seconds |
| `--cooldown-ticks` | `u32` | `1` | Ticks to skip after each actuation before next adjustment |
| `--tpot-limit` | `f64` | None | TPOT upper bound in ms. Activates AIMD. Requires `--stream` |
| `--tps-limit` | `f64` | None | TPS lower bound (tokens/sec). Activates AIMD. Requires `--stream` |
| `--all-tokens-limit` | `u64` | None | Total context tokens upper bound. Activates AIMD |

Without any constraint args (`--tpot-limit`, `--tps-limit`, `--all-tokens-limit`), the controller simply ramps to `--bs-limit` and holds steady (static concurrency).

## Token Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-tokens` | `u64` | None | Safety cap on output tokens. Applied only when the dataset has no `output_length` (e.g. `plaintext`). Ignored when trace provides explicit `output_length` |
| `--context-length` | `u64` | None | Model context window (tokens). When `input_length + output_length` exceeds this, `min_tokens` is omitted from request body to avoid server-side 400 errors |

## Streaming

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--stream` | `bool` | `false` | Enable SSE streaming. Required for `--tpot-limit` and `--tps-limit` |

## SLO & Timeout

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ttft-slo` | `f32` | `5.0` | TTFT SLO in seconds |
| `--tpot-slo` | `f32` | `0.06` | TPOT SLO in seconds |
| `--early-stop-error-threshold` | `u32` | None | Early stop when timeout request count exceeds this threshold |

Per-request timeout is computed as: `max(15, ttft_slo + tpot_slo * output_length)` seconds. If a request does not complete within this window, it is aborted and recorded as a timeout.

## Runtime

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--time-in-secs`, `-t` | `u64` | `60` | Maximum runtime duration in seconds |
| `--threads` | `usize` | `55` | Tokio worker threads |

## Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-path`, `-o` | `String` | `./log/output.jsonl` | Per-request output file (JSONL) |
| `--summary-path` | `String` | `<output-path>.summary.json` | Aggregate summary file (JSON) |
| `--metric-percentile` | `Vec<u32>` | `90,95,99` | Comma-separated percentiles to report for latency metrics |
| `--tracing-path` | `String` | None | Detailed tracing output file (only with `release-with-debug` API) |

## Platform / API-specific

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-name` | `String` | None | Model name. Required for `openai`, `sgl`, `aibrix` APIs |
| `--aibrix-route` | `String` | None | AIBrix routing strategy: `prefix-cache`, `prefix-cache-preble`, `throughput` |
| `--rid-source` | `String` | `none` | Request ID source: `none` (no rid) or `content-hash` (SHA256 of messages) |

## Cache (hashed mode only)

Pre-generate all prompts to avoid runtime `TokenSampler` latency.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cache` | `String` | `none` | Cache mode: `none`, `tmpfs`, `file` |
| `--cache-path` | `String` | *(mode-dependent)* | Custom cache file path. Defaults: tmpfs → `/dev/shm/request-sim-cache.bin`, file → `./request-sim-cache.bin` |

## Hashed Mode Only

These arguments are only available when built **without** `--features prompt-text-plain` (the default hashed build).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tokenizer` | `String` | *(required)* | Path to `tokenizer.json` |
| `--tokenizer-config` | `String` | *(required)* | Path to `tokenizer_config.json` |
| `--num-producer` | `usize` | `1` | Producer threads in `TokenSampler` (recommended: 16) |
| `--channel-capacity` | `usize` | `128` | Channel capacity between producers and consumers (recommended: 10240) |
