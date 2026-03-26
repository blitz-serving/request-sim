# Runbook: PD Disaggregation Test with request-sim

Test request-sim's three modes against a SGLang PD (Prefill-Decode) disaggregated deployment on master-h100.

## Setup

- **Model**: Qwen3-30B-A3B-Thinking-2507 (MoE, 3B active / 30B total)
- **Prefill**: GPU 0,1 — tp=2, ep=2, port 8002, bootstrap port 8998
- **Decode**: GPU 2,3 — tp=2, dp=2, ep=2, DP-attention, port 8001
- **Router**: sglang-router MiniLB, port 8003 (coordinates prefill + decode)
- **Transfer**: Mooncake IB backend (ib7s400p0–p3)
- **Trace**: `/workspace/traces/mmx-20260318-hour16-1-filtered50k.jsonl` (37 entries, input ≤ 50K tokens)

## Architecture

```
Client (request-sim / curl)
       │
       ▼
  sglang-router (port 8003, MiniLB)
       │
       ├──► Prefill (port 8002, GPU 0,1)
       │         │
       │         │ KV transfer via Mooncake IB
       │         ▼
       └──► Decode  (port 8001, GPU 2,3) ──► SSE response back to client
```

The router sends the **same modified request** to both prefill and decode simultaneously. It injects `bootstrap_host`, `bootstrap_port`, and `bootstrap_room` into the request body so decode knows where to fetch KV cache from prefill. The response streams from decode only.

**Clients must send requests to the router (port 8003), not directly to prefill or decode.**

## Step 0: Restart PD Stack

**Order matters.** Prefill must be ready before decode starts (decode connects to prefill's bootstrap port). Router starts last.

```bash
ssh master-h100

# Stop all three
sudo docker stop zdy-sglang-agent-router zdy-sglang-agent-decode zdy-sglang-agent-prefill

# Start prefill FIRST
sudo docker start zdy-sglang-agent-prefill

# Wait for prefill ready (~15-20s)
watch -n2 'sudo docker logs --tail 3 zdy-sglang-agent-prefill 2>&1'
# Look for: "The server is fired up and ready to roll!"
# Ctrl-C once you see it

# THEN start decode
sudo docker start zdy-sglang-agent-decode

# Wait for decode ready (~30-40s, cuda graph capture takes time)
watch -n2 'sudo docker logs --tail 3 zdy-sglang-agent-decode 2>&1'
# Look for: "End of prefill disaggregation mode warmup with status 200"
# Then:     "The server is fired up and ready to roll!"

# THEN start router
sudo docker start zdy-sglang-agent-router
```

### Container creation commands (if containers don't exist)

**Prefill** (GPU 0,1):
```bash
sudo docker run -d --name zdy-sglang-agent-prefill \
  --gpus '"device=0,1"' --network host --shm-size 32g --restart unless-stopped \
  -v /mnt/disk2/models:/nvme2/models -v /mnt/disk1/models:/nvme1/models \
  -v /mnt/disk4/agents/zdy/claude/master-workspace:/workspace \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path /nvme2/models/Qwen3-30B-A3B-Thinking-2507 \
    --reasoning-parser qwen3-thinking \
    --host 0.0.0.0 --port 8002 \
    --tp 2 --ep 2 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-transfer-backend mooncake \
    --mooncake-ib-device ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3
```

**Decode** (GPU 2,3):
```bash
sudo docker run -d --name zdy-sglang-agent-decode \
  --gpus '"device=2,3"' --network host --shm-size 32g --restart unless-stopped \
  -v /mnt/disk2/models:/nvme2/models -v /mnt/disk1/models:/nvme1/models \
  -v /mnt/disk4/agents/zdy/claude/master-workspace:/workspace \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path /nvme2/models/Qwen3-30B-A3B-Thinking-2507 \
    --reasoning-parser qwen3-thinking \
    --host 0.0.0.0 --port 8001 \
    --tp 2 --dp 2 --ep 2 --enable-dp-attention \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-transfer-backend mooncake \
    --mooncake-ib-device ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3
```

**Router** (no GPU):
```bash
sudo docker run -d --name zdy-sglang-agent-router \
  --network host --restart unless-stopped \
  -e http_proxy=http://localhost:11237 \
  -e https_proxy=http://localhost:11237 \
  -e all_proxy=socks5h://localhost:11236 \
  lmsysorg/sglang:latest \
  bash -c 'pip install sglang-router && python3 -m sglang_router.launch_router \
    --host 0.0.0.0 --port 8003 \
    --pd-disaggregation \
    --prefill http://127.0.0.1:8002 8998 \
    --decode http://127.0.0.1:8001 \
    --mini-lb'
```

## Step 1: Verify PD Works

**IMPORTANT**: This is a thinking model (Qwen3-30B-A3B-Thinking with `--reasoning-parser qwen3-thinking`). Even a trivial "hi" request generates many thinking tokens before responding, taking **60-90 seconds**. Always use `--max-time 300` or longer.

Send requests to the **router** (port 8003). The router handles `bootstrap_room`/`bootstrap_host` automatically.

```bash
curl -s --max-time 300 http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/nvme2/models/Qwen3-30B-A3B-Thinking-2507",
    "messages": [{"role": "user", "content": "Say hi in one word"}],
    "max_tokens": 20, "min_tokens": 5,
    "stream": true
  }'
```

**Interpreting results:**
- If you see SSE `data:` lines streaming back → PD works.
- If curl times out after 300s → the KV transfer is broken, redo Step 0.
- **Never use `--max-time` shorter than 120s** — the thinking model needs time. If curl times out mid-request, it can trigger a scheduler crash bug (SGLang `KeyError` → SIGQUIT).

## Step 2: Prepare request-sim

```bash
cd /workspace/request-sim

# Build (if not already built):
source ~/.cargo/env
cargo build --release --features prompt-text-plain -j64
```

## Step 3: Create Output Directories

```bash
mkdir -p log/feedback_sweep log/random_process log/trace_replay
```

## Step 4: Common Variables

```bash
ENDPOINT="http://localhost:8003/v1/chat/completions"
MODEL="/nvme2/models/Qwen3-30B-A3B-Thinking-2507"
TRACE="/workspace/traces/mmx-20260318-hour16-1-filtered50k.jsonl"
BIN="./target/release/request-sim"
```

**Note**: The endpoint is the **router** (port 8003), not prefill or decode directly. Do NOT use `--bootstrap-room` — the router sets it automatically.

## Step 5: Mode 1 — Feedback (tune target_bs for TPOT ≈ 35ms)

**Note**: TPOT tuning via `target_bs` sweep is currently blocked by Mooncake session instability (see Known Issues #4). At bs=1, TPOT ≈ 7-8ms. bs≥2 kills the PD stack.

```bash
$BIN \
  --mode feedback \
  --target-bs 1 \
  --dataset minimax \
  --dataset-path $TRACE \
  --api openai \
  --endpoint $ENDPOINT \
  --model-name $MODEL \
  --stream \
  --ttft-slo 120 \
  --tpot-slo 0.5 \
  --time-in-secs 120 \
  --output-path ./log/feedback_sweep/bs1.jsonl \
  --summary-path ./log/feedback_sweep/bs1.summary.json

  echo "--- Summary for bs=$BS ---"
  python3 -m json.tool ./log/feedback_sweep/bs${BS}.summary.json
  echo ""
done
```

## Step 6: Mode 2 — Random Process (Poisson)

```bash
$BIN \
  --mode random-process \
  --arrival poisson \
  --rate 0.1 \
  --dataset minimax \
  --dataset-path $TRACE \
  --api openai \
  --endpoint $ENDPOINT \
  --model-name $MODEL \
  --stream \
  --ttft-slo 120 \
  --tpot-slo 0.5 \
  --time-in-secs 90 \
  --output-path ./log/random_process/poisson_r01.jsonl \
  --summary-path ./log/random_process/poisson_r01.summary.json

python3 -m json.tool ./log/random_process/poisson_r01.summary.json
```

Keep `--rate` very low (≤0.1) to avoid concurrent PD requests that kill Mooncake sessions.

## Step 7: Mode 3 — Trace Replay

```bash
$BIN \
  --mode trace-replay \
  --scale-factor 3.0 \
  --dataset minimax \
  --dataset-path $TRACE \
  --api openai \
  --endpoint $ENDPOINT \
  --model-name $MODEL \
  --stream \
  --ttft-slo 120 \
  --tpot-slo 0.5 \
  --time-in-secs 300 \
  --output-path ./log/trace_replay/sf3.jsonl \
  --summary-path ./log/trace_replay/sf3.summary.json

python3 -m json.tool ./log/trace_replay/sf3.summary.json
```

Use `--scale-factor` to speed up (< 1.0) or slow down (> 1.0) the replay. Use higher values (≥3.0) to keep concurrent PD load low.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Curl returns empty after 30-60s | Thinking model needs 60-90s per request | Use `--max-time 300` |
| `KVTransferError: Aborted by AbortReq` | Client disconnected (curl timeout too short) | Increase curl/request timeout |
| `Scheduler hit an exception: KeyError` + SIGQUIT | SGLang bug: scheduler crash on request cleanup | Redo Step 0. Avoid short curl timeouts |
| `Request N timed out after 300.0s in KVPoll.Bootstrapping` | KV transfer hung (stale IB or poisoned scheduler) | Redo Step 0 |
| Requests hang forever | Mooncake IB channel broken | Redo Step 0 (restart all three, correct order) |
| `Failed to resolve 'none'` (decode log) | Decode received request without `bootstrap_host` | Send requests through the router (port 8003), not directly to prefill/decode |
| Prefill OOM / container exits | Input too long or concurrency too high | Use filtered trace, lower `target_bs` |
| `cargo: command not found` | Cargo not in PATH | `source ~/.cargo/env` |
| Container exited after OOM | GPU memory exhausted | `sudo docker start <name>`, but must redo Step 0 |
| Router port conflict (`[Errno 98]`) | Port 8003 already in use | Check `ss -tlnp | grep :8003`, kill or choose another port |
| Router pip install hangs | No network proxy in router container | Add proxy env vars when creating container |

## Test Results (2026-03-25)

Using synthetic 20-entry trace (`synth_20.jsonl`, input ~20 tokens, output 30-150 tokens):

| Mode | Config | Success | TPOT Mean | TPOT P90 | TTFT Mean | Throughput |
|------|--------|---------|-----------|----------|-----------|------------|
| Feedback | bs=1 | 18/20 | 7-8ms | ~31ms | 632ms | 64.5 tok/s |
| Random-process | Poisson r=0.1 | 11/11 | 18.9ms | 31.9ms | 705ms | 12 tok/s |
| Trace-replay | sf=3.0 | 10/20* | 14.1ms | 33.2ms | 596ms | 119 tok/s |

*PD degraded mid-run due to Mooncake session lifetime issue (see Known Issues #4).

**SLO timeout settings**: For thinking models, use `--ttft-slo 120 --tpot-slo 0.5` (default 5s/0.06s is too short).

**TPOT tuning blocked**: bs≥2 immediately kills Mooncake sessions → all requests abort. Cannot sweep target_bs for TPOT≈35ms.

## Known Issues

1. **Thinking model latency**: Qwen3-30B-A3B-Thinking generates extensive `<think>...</think>` tokens before the visible response. A simple "hi" request can take 60-90 seconds. This is normal behavior, not a PD issue.

2. **SGLang scheduler crash on request cleanup (KeyError → SIGQUIT)**: When a request's KV transfer times out or is aborted, the scheduler's cleanup code crashes with `KeyError: N` → SIGQUIT kills the server. This is a bug in SGLang 0.5.9's PD disaggregation error handling. Avoid triggering it by using sufficiently long timeouts.

3. **sglang-router IS required**: The PD setup requires the sglang-router (MiniLB) to coordinate requests. The router sends the same request to both prefill and decode simultaneously, injecting `bootstrap_host` and `bootstrap_port` so decode knows where to fetch KV cache. Without the router, decode never receives the request and prefill's KV transfer times out after 300s.

4. **Mooncake IB session lifetime**: Mooncake IB/TCP transfer sessions degrade and die after ~15-20 sequential requests. With concurrent requests (bs≥2), sessions die immediately. Symptoms: `KVTransferError: Failed to get kvcache from prefill instance, it might be dead` and `TcpTransport::startTransfer: connect: Cannot assign requested address`. **Root cause**: likely a resource leak or session management bug in Mooncake transport. **Workaround**: restart the full PD stack (Step 0) when sessions die. Use low request rates (≤0.1 req/s) and bs=1 to maximize session lifetime.

5. **request-sim `--bootstrap-room` not needed with router**: When using the router, do NOT pass `--bootstrap-room` to request-sim. The router automatically generates and injects `bootstrap_room` into each request.

6. **50K-token trace too large for PD**: The production trace (`mmx-20260318-hour16-1-filtered50k.jsonl`, 37 entries, input 1K-50K tokens) overwhelms the Mooncake transfer. Large prefills (20-50K tokens) stress the IB sessions. Use shorter synthetic traces for PD testing.
