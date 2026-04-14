# request-sim

Rust benchmark client for LLM inference endpoints. Replays production traces, generates stochastic workloads, or runs closed-loop concurrency sweeps — measuring TTFT, TPOT, and throughput.

Can achieve **100+ QPS and 500,000+ tokens/s** with ~30 CPU threads, sufficient for stress testing a 16/32-instance Qwen3-30B-A3B deployment.

> **Note**: In hashed mode (default), generated prompts are semantically meaningless — reconstructed from block hashes for KVCache hit pattern preservation. Only suitable for performance testing, not semantic evaluation.

> **Recommended**: Use the same `tokenizers` library version as the inference framework to ensure identical token counts.


## Features

- **Three request dispatch modes**: trace replay, stochastic arrival (Poisson/uniform), AIMD feedback controller
- Construct prompts from block hashes (preserving KVCache hit patterns) or plain text
- OpenAI, TGI, SGLang, and AIBrix API support
- SSE streaming with per-token latency tracking
- JSONL output with percentile summary
- High throughput, low resource usage (request timestamp drift < 5ms)

For the full CLI reference, see [`docs/arguments.md`](docs/arguments.md).


## Quick Start

### Build

```bash
# Hashed mode (default) — synthetic prompts from block hashes
cargo build --release -j64

# Plain text mode — meaningful prompts
cargo build --release --features prompt-text-plain -j64
```

### Run

```bash
# Trace replay (default mode)
./target/release/request-sim \
  --mode trace-replay \
  --scale-factor 1.5 \
  --dataset bailian --dataset-path trace.jsonl \
  --tokenizer tokenizer.json --tokenizer-config tokenizer_config.json \
  --api openai --endpoint http://localhost:8080/v1/chat/completions \
  --model-name Qwen2.5-7B-Instruct \
  --time-in-secs 600

# Poisson arrivals at 10 req/s
./target/release/request-sim \
  --mode random-process \
  --arrival poisson --rate 10.0 \
  --dataset bailian --dataset-path trace.jsonl \
  --tokenizer tokenizer.json --tokenizer-config tokenizer_config.json \
  --api openai --endpoint http://localhost:8080/v1/chat/completions \
  --model-name Qwen2.5-7B-Instruct \
  --time-in-secs 120

# AIMD feedback: probe up to 64 concurrent, retreat if TPOT > 50ms
./target/release/request-sim \
  --mode feedback \
  --bs-limit 64 --tpot-limit 50 \
  --stream \
  --dataset bailian --dataset-path trace.jsonl \
  --tokenizer tokenizer.json --tokenizer-config tokenizer_config.json \
  --api sgl --endpoint http://localhost:8080/v1/chat/completions \
  --model-name Qwen2.5-7B-Instruct \
  --time-in-secs 300

# Plain text mode (requires --features prompt-text-plain build)
./target/release/request-sim \
  --mode feedback --bs-limit 4 \
  --dataset plaintext --dataset-path prompts.txt \
  --api sgl --endpoint http://localhost:8080/v1/chat/completions \
  --model-name Qwen2.5-7B-Instruct \
  --max-tokens 4096 --stream \
  --time-in-secs 300
```


## Request Modes

- **trace-replay** (default): replays at original trace timestamps scaled by `--scale-factor`. Terminates when dataset is exhausted or `--time-in-secs`.
- **random-process**: stochastic inter-arrival times (`--arrival poisson|uniform`, `--rate` req/s). Cycles dataset indefinitely; terminates on `--time-in-secs` only.
- **feedback**: AIMD closed-loop controller probes concurrency from 1 toward `--bs-limit`, retreating when constraints (`--tpot-limit`, `--tps-limit`, `--all-tokens-limit`) are violated.


## Prompt Text Modes

Two compile-time modes control prompt construction:

| Build | Datasets | Prompt content |
|-------|----------|---------------|
| Default (hashed) | `bailian`, `mooncake` | Synthetic noise from block hashes via `TokenSampler` |
| `--features prompt-text-plain` | `plaintext`, `openai` | Meaningful real text; model generates to EOS |

Use `--max-tokens` to cap output length when the dataset doesn't specify one (e.g. `plaintext`).


## Supported APIs

| API | Flag | Endpoint format |
|-----|------|-----------------|
| OpenAI | `--api openai` | `/v1/chat/completions` |
| SGLang | `--api sgl` | `/v1/chat/completions` (+ usage/cached_tokens extraction) |
| TGI | `--api tgi` | `/generate` |
| AIBrix | `--api aibrix` | `/v1/chat/completions` (+ routing strategy header) |


## Output Format

Results are written to the JSONL file specified by `--output-path`. Each line contains:

| Field | Description |
|-------|-------------|
| `status` | HTTP status code |
| `s_time_drift` | Deviation between scheduled and actual send time (ms, expect < 5ms) |
| `s_time` / `e_time` | Send and end timestamps |
| `input_length` / `output_length` | Token counts |
| `span_time` | End-to-end latency (ms) |

Additional fields vary by API (TTFT, TPOT, etc.). A percentile summary is written to `--summary-path`.


## Documentation

- [`docs/arguments.md`](docs/arguments.md) — Complete CLI reference
- [`docs/extending.md`](docs/extending.md) — How to add new API backends


## Sponsor

This project is sponsored by **Alibaba Tongyi Lab**.

## Contact

For technical questions, bug reports, and feature requests, please use
GitHub [Issues](https://github.com/blitz-serving/trace-replayer/issues).

For other inquiries, you may contact the maintainers via GitHub ([Healthcliff-Ding](https://github.com/Healthcliff-Ding)).
