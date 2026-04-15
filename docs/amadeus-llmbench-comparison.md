# Amadeus Implementation: llm-bench (Python) → request-sim (Rust) Mapping

This document maps the key code paths between llm-bench's amadeus backend and
request-sim's port, for human review of equivalence.

## 1. Dataset Loading

### llm-bench: `datasets/amadeus_replay.py` → `sample_amadeus_replay_requests()`

```python
data = json.loads(line)
chat_id = data.get("chat_id", f"line_{line_idx}")
conversation = data.get("data", [])
model_control = data.get("model_control", {})
output_len = model_control.get("tokens_to_generate", 256)
total_chars = sum(len(turn.get("text", "")) for turn in conversation)
prompt_len = total_chars // 3

quest_code = data.get("quest_code") or "talkie_abab55"
original_body = {
    "data": conversation,
    "quest_code": quest_code,
    "model_control": model_control,
}
if "system_data" in data:
    original_body["system_data"] = data.get("system_data")
for key in ["functions", "function_call", "tools"]:
    if key in data:
        original_body[key] = data.get(key)

timestamp = data.get("timestamp", None)
if timestamp is not None:
    timestamp = int(timestamp) if isinstance(timestamp, str) else timestamp

prompt = json.dumps(original_body)  # stored as serialized JSON string
```

### request-sim: `dataset.rs` → `AmadeusDataset::load()`

```rust
let raw: serde_json::Value = serde_json::from_str(&line)?;

let conversation = raw.get("data").and_then(|v| v.as_array());
let total_chars: usize = conversation.map(|arr| {
    arr.iter()
        .filter_map(|turn| turn.get("text").and_then(|t| t.as_str()))
        .map(|s| s.len()).sum()
}).unwrap_or(0);
let input_length = (total_chars / 3) as u64;

let output_length = raw.get("model_control")
    .and_then(|mc| mc.get("tokens_to_generate"))
    .and_then(|v| v.as_u64())
    .unwrap_or(256);

let timestamp_ms = raw.get("timestamp").map(|v| match v {
    Value::Number(n) => n.as_u64().unwrap_or(0),
    Value::String(s) => s.parse::<u64>().unwrap_or(0),
    _ => 0,
}).unwrap_or(lineno as u64);

let quest_code = raw.get("quest_code")
    .and_then(|v| v.as_str())
    .filter(|s| !s.is_empty())
    .unwrap_or("talkie_abab55");

// Build body preserving: data, quest_code, model_control, system_data, functions, function_call, tools
// Stored as serde_json::Value (not serialized string)
```

### Key differences

| Aspect | llm-bench | request-sim |
|--------|-----------|-------------|
| Storage | `json.dumps(body)` → string in `DatasetRow.prompt` | `serde_json::Value` in `AmadeusItem.body` |
| chat_id prefix | Adds random UUID prefix to avoid session collisions | Not implemented (no session concept) |
| Sampling | Supports `num_requests` cap, session-aware sampling | Loads all lines |

---

## 2. Request Body Construction

### llm-bench: `backends/amadeus.py` → `_do_single_daodao_request()`

```python
# When prompt already has quest_code (amadeus-replay data):
body = json.loads(request_func_input.prompt)
body = _prepare_amadeus_body(body, request_func_input, trace_id, is_converted=False)
#   → body['trace_id'] = trace_id
#   → body['quest_code'] = body.get('quest_code') or 'talkie_abab55'

data_json = json.dumps(body, ensure_ascii=False)

amadeus_data = {
    "data_list": [
        {"data_value": data_json, "data_type": 1}
    ],
    "config": None,
    "meta": {"amadeus_id": amadeus_id},
    "options": {"stream": not disable_stream, "skip_info_mask": True},
}

headers = {"content-type": "application/json", "trace-id": trace_id}
```

### request-sim: `apis/amadeus_api.rs` → `AmadeusApi::request_json_body()`

```rust
let mut inner_body = match prompt {
    PromptPayload::Body(val) => val,
    // ...
};

let trace_id = uuid::Uuid::new_v4().to_string().replace("-", "");
inner_body["trace_id"] = json!(trace_id);

let data_json = serde_json::to_string(&inner_body).unwrap();
let amadeus_id = AMADEUS_ID.get().and_then(|opt| *opt);

let envelope = json!({
    "data_list": [{"data_value": data_json, "data_type": 1}],
    "meta": {"amadeus_id": amadeus_id, "request_id": trace_id},
    "options": {"stream": stream, "skip_info_mask": true},
});

// extra_headers() extracts trace_id from meta.request_id → ("trace-id", trace_id)
```

### Key differences

| Aspect | llm-bench | request-sim |
|--------|-----------|-------------|
| `"config"` field | `"config": None` (present) | Absent (omitted from JSON) |
| `meta` fields | `{"amadeus_id": ...}` only | `{"amadeus_id": ..., "request_id": ...}` (extra field for header extraction) |
| trace_id generation | `uuid.uuid4().hex` | `Uuid::new_v4().to_string().replace("-","")` |
| `ensure_ascii` | `json.dumps(body, ensure_ascii=False)` | `serde_json::to_string` (UTF-8 passthrough, no ASCII escaping) |
| `_prepare_amadeus_body` | Merges `extra_request_body["model_control"]`, adds `chat_id` from session | Not implemented (body used as-is from dataset) |

---

## 3. Streaming Response Parsing

### llm-bench: `backends/amadeus.py` → streaming section of `_do_single_daodao_request()`

```python
async for line in response.content:
    line = line.strip()
    if not line: continue
    if line.startswith(b"data:"):
        chunk_data = json.loads(line[5:])  # outer envelope

        if 'status_code' in chunk_data and chunk_data['status_code'] != 0:
            # error
            break

        if 'data' not in chunk_data or not chunk_data['data']:
            continue

        base_reply = json.loads(chunk_data["data"])  # inner JSON (string → dict)

        if "output_tokens_count" in base_reply:
            output_tokens_count = base_reply["output_tokens_count"]
        if "uncached_input_tokens_count" in base_reply:
            output.uncached_prompt_len = base_reply["uncached_input_tokens_count"]

        timestamp = time.perf_counter()
        if ttft == 0.0:
            ttft = timestamp - st
        else:
            output.itl.append(timestamp - most_recent_timestamp)

        generated_text += base_reply.get("text", "") + base_reply.get("reasoning_content", "")

# After loop:
output.output_len = output_tokens_count + 1  # +1 for EOS
```

### request-sim: `apis/amadeus_api.rs` → `parse_response()` streaming section

```rust
if trimmed.starts_with("data:") {
    let data_str = trimmed[5..].trim();
    if let Ok(chunk) = serde_json::from_str::<Value>(data_str) {
        // Check status_code
        if chunk["status_code"].as_i64() is Some(code) && code != 0 { continue; }

        // Parse inner JSON from "data" field
        if let Some(inner_str) = chunk.get("data").and_then(|d| d.as_str()) {
            if let Ok(inner) = serde_json::from_str::<Value>(inner_str) {
                // Update cumulative output_tokens_count
                if let Some(otc) = inner["output_tokens_count"].as_u64() {
                    last_output_tokens = otc;
                }
                // Extract uncached_input_tokens_count
                ...
                // Token event if text or reasoning_content non-empty
                let text = inner["text"].as_str().unwrap_or("");
                let reasoning = inner["reasoning_content"].as_str().unwrap_or("");
                if !text.is_empty() || !reasoning.is_empty() {
                    token_count += 1;
                    // TTFT / TBT tracking
                }
            }
        }
    }
}
// After loop:
// output_length = last_output_tokens + 1  (+1 for EOS)
```

### Key differences

| Aspect | llm-bench | request-sim |
|--------|-----------|-------------|
| Token counting | Every chunk with `data` field counts as a token event (TTFT/ITL) | Only chunks with non-empty `text` or `reasoning_content` |
| Error handling | `break` on error status | `continue` (skips error chunk, keeps reading) |
| `generated_text` | Accumulated for success check | Not accumulated (request-sim uses token_count only) |
| `function_calls` | Parsed and accumulated | Not parsed |
| Output tokens | `output_tokens_count + 1` (+1 EOS per beam) | `last_output_tokens + 1` (same, single beam only) |
| Multi-beam | Supported via `text_infos[]` loop | Single beam only |

---

## 4. Non-Streaming Response Parsing

### llm-bench

```python
response_json = await response.json()
base_resp = response_json.get("base_resp", {})
if base_resp.get("status_code", 0) != 0:
    # error
else:
    resp = json.loads(response_json["data_list"][0]["data_value"])
    # Extract from text_infos[]:
    #   total_output_tokens = sum(ti["output_tokens_count"])
    #   output.output_len = total_output_tokens + num_beams  (+1 per beam for EOS)
    #   output.uncached_prompt_len = text_infos[-1]["uncached_input_tokens_count"]
```

### request-sim

```rust
let v = serde_json::from_str::<Value>(&body)?;
let status_ok = v["base_resp"]["status_code"].as_i64().map(|c| c == 0).unwrap_or(true);
if status_ok {
    let data_value = v["data_list"][0]["data_value"].as_str()?;
    let resp = serde_json::from_str::<Value>(data_value)?;
    // Sum text_infos[].output_tokens_count + num_beams
    // Extract uncached_input_tokens_count from last text_info
}
```

### Key differences

| Aspect | llm-bench | request-sim |
|--------|-----------|-------------|
| `generated_text` extraction | Extracts `reply`, `text_infos[].reply`, `reasoning_content` | Not extracted |
| `function_calls` | Parsed from response | Not parsed |
| Effective output len | Tracks max beam separately | Not tracked |

---

## 5. Known Divergences Summary

1. **`"config": null`** — llm-bench includes it in the envelope; request-sim omits it. The server ignores null values, so this is functionally equivalent.

2. **`meta.request_id`** — request-sim adds this extra field so `extra_headers()` can extract the trace-id without re-parsing nested `data_value`. The server ignores unknown meta fields.

3. **Session / chat_id** — llm-bench prepends a random UUID prefix to chat_id to avoid cross-run collisions. request-sim does not implement sessions.

4. **model_control merging** — llm-bench's `_prepare_amadeus_body` can merge CLI-provided model_control overrides into the body. request-sim uses the body as-is from the dataset.

5. **Token event definition** — llm-bench counts every `data` chunk as a token event; request-sim only counts chunks with non-empty text/reasoning_content. The latter is more precise.

6. **Error on streaming** — llm-bench breaks the loop on server error; request-sim continues. This means request-sim may report partial results from error responses.

7. **Multi-beam** — llm-bench handles multiple beams in `text_infos[]`; request-sim sums all beams (non-streaming) but doesn't track effective_output_len per beam.

8. **function_calls / tool_calls** — Not implemented in request-sim. llm-bench accumulates streaming tool call deltas.
