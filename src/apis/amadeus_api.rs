use super::{InFlightState, LLMApi, METRIC_PERCENTILES, RequestError};
use crate::dataset::PromptPayload;
use futures_util::TryStreamExt;
use reqwest::Response;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    time::{timeout as tokio_timeout, Instant as TokioInstant},
};
use tokio_util::io::StreamReader;

#[derive(Copy, Clone)]
pub struct AmadeusApi;

/// Optional amadeus_id injected into envelope `meta`.
pub static AMADEUS_ID: OnceLock<Option<i64>> = OnceLock::new();

const DEFAULT_PERCENTILES: [u32; 3] = [90, 95, 99];

#[async_trait::async_trait]
impl LLMApi for AmadeusApi {
    fn request_json_body(
        prompt: PromptPayload,
        _input_length: u64,
        _output_length: u64,
        stream: bool,
    ) -> String {
        let mut inner_body = match prompt {
            PromptPayload::Body(val) => val,
            PromptPayload::Content(text) => {
                // Wrap plain text in minimal amadeus body
                json!({
                    "data": [{"role": "user", "text": text}],
                    "quest_code": "talkie_abab55",
                })
            }
            PromptPayload::Messages(_) => {
                panic!("Messages payload not supported for Amadeus API; use --dataset amadeus-replay")
            }
        };

        // Generate per-request trace_id
        let trace_id = uuid::Uuid::new_v4().to_string().replace("-", "");

        // Add trace_id to inner body
        if let Some(obj) = inner_body.as_object_mut() {
            obj.insert("trace_id".to_string(), json!(trace_id));
        }

        // Build amadeus envelope
        let data_json = serde_json::to_string(&inner_body).unwrap();
        let amadeus_id = AMADEUS_ID.get().and_then(|opt| *opt);

        let envelope = json!({
            "data_list": [{"data_value": data_json, "data_type": 1}],
            "meta": {
                "amadeus_id": amadeus_id,
                "request_id": trace_id,
            },
            "options": {
                "stream": stream,
                "skip_info_mask": true,
            },
        });

        envelope.to_string()
    }

    fn extra_headers(body: &str) -> Vec<(&'static str, String)> {
        // Extract trace_id from meta.request_id for the trace-id HTTP header
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
            if let Some(tid) = v
                .get("meta")
                .and_then(|m| m.get("request_id"))
                .and_then(|v| v.as_str())
            {
                return vec![("trace-id", tid.to_string())];
            }
        }
        vec![]
    }

    async fn parse_response(
        response: Response,
        stream: bool,
        timeout_duration: Duration,
        in_flight: Option<Arc<InFlightState>>,
    ) -> Result<BTreeMap<String, String>, RequestError> {
        let mut result = BTreeMap::new();
        result.insert("status".to_string(), response.status().as_str().to_string());

        if !stream {
            // Non-streaming: parse full response
            if response.status().is_success() {
                if let Ok(body) = response.text().await {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                        // Check base_resp status
                        let status_ok = v
                            .get("base_resp")
                            .and_then(|br| br.get("status_code"))
                            .and_then(|sc| sc.as_i64())
                            .map(|code| code == 0)
                            .unwrap_or(true);

                        if status_ok {
                            if let Some(data_list) = v.get("data_list").and_then(|dl| dl.as_array())
                            {
                                if let Some(first) = data_list.first() {
                                    if let Some(data_value) =
                                        first.get("data_value").and_then(|dv| dv.as_str())
                                    {
                                        if let Ok(resp) =
                                            serde_json::from_str::<serde_json::Value>(data_value)
                                        {
                                            // Extract output token count from text_infos
                                            if let Some(text_infos) = resp
                                                .get("text_infos")
                                                .and_then(|ti| ti.as_array())
                                            {
                                                let total_output: u64 = text_infos
                                                    .iter()
                                                    .filter_map(|ti| {
                                                        ti.get("output_tokens_count")
                                                            .and_then(|v| v.as_u64())
                                                    })
                                                    .sum();
                                                let num_beams = text_infos.len() as u64;
                                                // +1 per beam for EOS token (matches llm-bench)
                                                if total_output > 0 {
                                                    result.insert(
                                                        "output_length".to_string(),
                                                        (total_output + num_beams).to_string(),
                                                    );
                                                }
                                                // uncached input tokens from last beam
                                                if let Some(last) = text_infos.last() {
                                                    if let Some(uncached) = last
                                                        .get("uncached_input_tokens_count")
                                                        .and_then(|v| v.as_u64())
                                                    {
                                                        result.insert(
                                                            "uncached_prompt_tokens".to_string(),
                                                            uncached.to_string(),
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return Ok(result);
        }

        // Streaming response handling
        if !response.status().is_success() {
            return Ok(result);
        }

        let byte_stream = response.bytes_stream();
        let stream_reader = StreamReader::new(
            byte_stream.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)),
        );
        let mut reader = BufReader::new(stream_reader);
        let mut line = String::new();
        let mut first_token_time: Option<TokioInstant> = None;
        let mut last_token_time: Option<TokioInstant> = None;
        let mut token_count: u64 = 0;
        let mut tbt_values: Vec<f64> = Vec::new();
        let mut tbt_except_first: Vec<f64> = Vec::new();
        let start_time = TokioInstant::now();

        // Accumulate generated text for output tracking
        let mut output_text = String::new();

        // Track cumulative output_tokens_count from chunks
        let mut last_output_tokens: u64 = 0;

        loop {
            if start_time.elapsed() > timeout_duration {
                return Err(RequestError::Timeout);
            }
            let remaining_duration = timeout_duration - start_time.elapsed();

            let read_future = reader.read_line(&mut line);
            match tokio_timeout(remaining_duration, read_future).await {
                Ok(Ok(0)) => break, // EOF
                Ok(Ok(_)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        line.clear();
                        continue;
                    }

                    if trimmed.starts_with("data:") {
                        let data_str = trimmed[5..].trim();

                        // Parse outer amadeus SSE chunk
                        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data_str) {
                            // Check for server error
                            if let Some(status_code) =
                                chunk.get("status_code").and_then(|sc| sc.as_i64())
                            {
                                if status_code != 0 {
                                    line.clear();
                                    continue;
                                }
                            }

                            // Parse inner JSON from "data" field (JSON-encoded string)
                            if let Some(inner_str) =
                                chunk.get("data").and_then(|d| d.as_str())
                            {
                                if inner_str.is_empty() {
                                    line.clear();
                                    continue;
                                }

                                if let Ok(inner) =
                                    serde_json::from_str::<serde_json::Value>(inner_str)
                                {
                                    // Update output_tokens_count
                                    if let Some(otc) = inner
                                        .get("output_tokens_count")
                                        .and_then(|v| v.as_u64())
                                    {
                                        last_output_tokens = otc;
                                    }

                                    // Extract uncached_input_tokens_count
                                    if let Some(uncached) = inner
                                        .get("uncached_input_tokens_count")
                                        .and_then(|v| v.as_u64())
                                    {
                                        result.insert(
                                            "uncached_prompt_tokens".to_string(),
                                            uncached.to_string(),
                                        );
                                    }

                                    // Check if this chunk has content (text or reasoning)
                                    let text = inner
                                        .get("text")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    let reasoning = inner
                                        .get("reasoning_content")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");

                                    if !text.is_empty() || !reasoning.is_empty() {
                                        output_text.push_str(text);
                                        output_text.push_str(reasoning);

                                        let now = TokioInstant::now();
                                        token_count += 1;

                                        // Update in-flight state
                                        if let Some(ref state) = in_flight {
                                            state
                                                .output_tokens_so_far
                                                .fetch_add(1, Ordering::Relaxed);
                                            if token_count == 1 {
                                                let ts = crate::get_timestamp() as u64;
                                                let _ = state
                                                    .first_token_time_ms
                                                    .compare_exchange(
                                                        0,
                                                        ts,
                                                        Ordering::Release,
                                                        Ordering::Relaxed,
                                                    );
                                            }
                                        }

                                        if first_token_time.is_none() {
                                            first_token_time = Some(now);
                                            let first_token_duration = now
                                                .duration_since(start_time)
                                                .as_secs_f64()
                                                * 1000.0;
                                            result.insert(
                                                "first_token_time".to_string(),
                                                format!("{first_token_duration:.3}"),
                                            );
                                        } else if let Some(last) = last_token_time {
                                            let tbt =
                                                now.duration_since(last).as_secs_f64() * 1000.0;
                                            tbt_values.push(tbt);
                                            if token_count > 2 {
                                                tbt_except_first.push(tbt);
                                            }
                                        }

                                        last_token_time = Some(now);
                                    }
                                }
                            }
                        }
                    }
                    line.clear();
                }
                Ok(Err(e)) => return Err(RequestError::StreamErr(e)),
                Err(_) => return Err(RequestError::Timeout),
            }
        }

        // Final output token count (+1 for EOS, matching llm-bench convention)
        if last_output_tokens > 0 {
            result.insert(
                "output_length".to_string(),
                (last_output_tokens + 1).to_string(),
            );
        }

        if let Some(first) = first_token_time {
            if let Some(last) = last_token_time {
                let total_time = last.duration_since(first).as_secs_f64() * 1000.0;
                result.insert("total_time".to_string(), format!("{total_time:.3}"));
            }
        }
        result.insert("token_count".to_string(), token_count.to_string());

        if !output_text.is_empty() {
            result.insert("output_text".to_string(), output_text);
        }

        // TBT statistics (same pattern as sgl_api.rs)
        if !tbt_except_first.is_empty() {
            let max_tbt_except_first = tbt_except_first
                .iter()
                .copied()
                .fold(f64::MIN, f64::max);
            result.insert(
                "max_time_between_tokens_except_first".to_string(),
                format!("{max_tbt_except_first:.3}"),
            );
        }

        if !tbt_values.is_empty() {
            let max_tbt = tbt_values.iter().copied().fold(f64::MIN, f64::max);
            result.insert(
                "max_time_between_tokens".to_string(),
                format!("{max_tbt:.3}"),
            );
            let avg_tbt = tbt_values.iter().sum::<f64>() / tbt_values.len() as f64;
            result.insert(
                "avg_time_between_tokens".to_string(),
                format!("{avg_tbt:.3}"),
            );
        }

        if !tbt_values.is_empty() {
            let mut sorted_tbt = tbt_values.clone();
            sorted_tbt
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let len = sorted_tbt.len();
            let percentiles = METRIC_PERCENTILES
                .get()
                .map(|v| v.as_slice())
                .unwrap_or(&DEFAULT_PERCENTILES);
            for percentile in percentiles {
                let idx = (len as f64 * (*percentile as f64 / 100.0)).ceil() as isize - 1;
                let idx = idx.max(0) as usize;
                let idx = idx.min(len - 1);
                result.insert(
                    format!("p{percentile}_time_between_tokens"),
                    format!("{:.3}", sorted_tbt[idx]),
                );
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::PromptPayload;

    /// Verify envelope structure matches llm-bench's amadeus backend.
    ///
    /// Reference: llm-bench/backends/amadeus.py, _do_single_daodao_request()
    /// ```python
    /// amadeus_data = {
    ///     "data_list": [{"data_value": data_json, "data_type": 1}],
    ///     "config": None,
    ///     "meta": {"amadeus_id": amadeus_id},
    ///     "options": {"stream": not disable_stream, "skip_info_mask": True},
    /// }
    /// ```
    #[test]
    fn envelope_structure_matches_llm_bench() {
        // Simulate an amadeus-replay body (as AmadeusDataset would produce)
        let inner = json!({
            "data": [{"role": "user", "text": "hello"}],
            "quest_code": "talkie_abab55",
            "model_control": {"tokens_to_generate": 128},
        });

        let body_str = AmadeusApi::request_json_body(
            PromptPayload::Body(inner.clone()),
            100, // input_length (unused by amadeus)
            128, // output_length (unused by amadeus)
            true,
        );

        let envelope: serde_json::Value = serde_json::from_str(&body_str).unwrap();

        // Top-level keys
        assert!(envelope.get("data_list").is_some(), "missing data_list");
        assert!(envelope.get("meta").is_some(), "missing meta");
        assert!(envelope.get("options").is_some(), "missing options");

        // data_list: array of 1 element with data_value (string) and data_type (1)
        let data_list = envelope["data_list"].as_array().unwrap();
        assert_eq!(data_list.len(), 1);
        assert_eq!(data_list[0]["data_type"], 1);
        let data_value_str = data_list[0]["data_value"].as_str().unwrap();

        // data_value is a JSON-encoded string of the inner body
        let data_value: serde_json::Value = serde_json::from_str(data_value_str).unwrap();
        assert_eq!(data_value["quest_code"], "talkie_abab55");
        assert!(data_value.get("data").is_some());
        assert!(data_value.get("model_control").is_some());

        // trace_id injected into inner body (llm-bench: body['trace_id'] = trace_id)
        let trace_id = data_value["trace_id"].as_str().unwrap();
        assert_eq!(trace_id.len(), 32, "trace_id should be 32-char hex UUID");

        // meta.request_id matches inner body trace_id
        // (llm-bench uses meta.amadeus_id only; request-sim also stores request_id for header extraction)
        assert_eq!(envelope["meta"]["request_id"].as_str().unwrap(), trace_id);

        // options
        assert_eq!(envelope["options"]["stream"], true);
        assert_eq!(envelope["options"]["skip_info_mask"], true);
    }

    /// Verify non-streaming envelope sets stream=false.
    #[test]
    fn envelope_stream_false() {
        let inner = json!({"data": [{"role": "user", "text": "hi"}], "quest_code": "test"});
        let body_str = AmadeusApi::request_json_body(
            PromptPayload::Body(inner),
            0, 0, false,
        );
        let envelope: serde_json::Value = serde_json::from_str(&body_str).unwrap();
        assert_eq!(envelope["options"]["stream"], false);
    }

    /// Verify Content prompt gets wrapped in minimal amadeus body.
    #[test]
    fn content_prompt_wrapping() {
        let body_str = AmadeusApi::request_json_body(
            PromptPayload::Content("hello world".to_string()),
            0, 0, true,
        );
        let envelope: serde_json::Value = serde_json::from_str(&body_str).unwrap();
        let data_value_str = envelope["data_list"][0]["data_value"].as_str().unwrap();
        let inner: serde_json::Value = serde_json::from_str(data_value_str).unwrap();

        // Should have data array with user message
        let data = inner["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["role"], "user");
        assert_eq!(data[0]["text"], "hello world");
        assert_eq!(inner["quest_code"], "talkie_abab55");
    }

    /// Verify extra_headers extracts trace-id from envelope.
    #[test]
    fn extra_headers_extracts_trace_id() {
        let inner = json!({"data": [], "quest_code": "test"});
        let body_str = AmadeusApi::request_json_body(
            PromptPayload::Body(inner),
            0, 0, true,
        );
        let headers = AmadeusApi::extra_headers(&body_str);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "trace-id");
        assert_eq!(headers[0].1.len(), 32);
    }

    /// Verify quest_code and optional fields are preserved from dataset.
    ///
    /// Reference: llm-bench/datasets/amadeus_replay.py
    /// ```python
    /// original_body = {
    ///     "data": conversation,
    ///     "quest_code": quest_code,
    ///     "model_control": model_control,
    /// }
    /// if "system_data" in data:
    ///     original_body["system_data"] = data.get("system_data")
    /// if "functions" in data:
    ///     original_body["functions"] = data.get("functions")
    /// ```
    #[test]
    fn optional_fields_preserved() {
        let inner = json!({
            "data": [{"role": "user", "text": "hi"}],
            "quest_code": "custom_quest",
            "model_control": {"tokens_to_generate": 64},
            "system_data": [{"text": "You are helpful"}],
            "functions": [{"name": "search"}],
            "tools": [{"type": "function"}],
        });

        let body_str = AmadeusApi::request_json_body(
            PromptPayload::Body(inner),
            0, 0, true,
        );
        let envelope: serde_json::Value = serde_json::from_str(&body_str).unwrap();
        let data_value_str = envelope["data_list"][0]["data_value"].as_str().unwrap();
        let dv: serde_json::Value = serde_json::from_str(data_value_str).unwrap();

        assert_eq!(dv["quest_code"], "custom_quest");
        assert!(dv.get("system_data").is_some());
        assert!(dv.get("functions").is_some());
        assert!(dv.get("tools").is_some());
    }

    // ---- Divergences from llm-bench (documented) ----
    //
    // 1. llm-bench envelope has "config": null — request-sim omits it (null fields
    //    are semantically absent; the server does not require it).
    //
    // 2. llm-bench meta has only "amadeus_id"; request-sim adds "request_id" to
    //    enable trace-id header extraction via extra_headers() without re-parsing
    //    the nested data_value JSON.
    //
    // 3. llm-bench generates trace_id = uuid.uuid4().hex (Python); request-sim
    //    uses uuid::Uuid::new_v4().to_string().replace("-", "") (Rust). Both
    //    produce 32-char lowercase hex strings.
}
