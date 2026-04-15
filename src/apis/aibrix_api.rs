use super::{InFlightState, LLMApi, RequestError, CONTEXT_LENGTH, MAX_TOKENS_CAP, MODEL_NAME};
use crate::dataset::PromptPayload;
use reqwest::Response;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

#[derive(Copy, Clone)]
pub struct AbxApi;

pub static AIBRIX_ROUTE_STRATEGY: OnceLock<String> = OnceLock::new();

#[async_trait::async_trait]
impl LLMApi for AbxApi {
    fn request_json_body(prompt: PromptPayload, input_length: u64, output_length: u64, stream: bool) -> String {
        let messages = match prompt {
            PromptPayload::Content(text) => json!([{"role": "user", "content": text}]),
            PromptPayload::Messages(val) => val,
            PromptPayload::Body(_) => panic!("Body payload not supported for AIBrix API"),
        };
        let mut body = json!({
            "model": MODEL_NAME.get().unwrap().as_str(),
            "messages": messages,
            "stream": stream,
        });
        if output_length > 0 {
            let exceeds_ctx = CONTEXT_LENGTH.get()
                .and_then(|opt| opt.as_ref())
                .is_some_and(|ctx| input_length + output_length > *ctx);
            if !exceeds_ctx {
                body["min_tokens"] = json!(output_length);
            }
            body["max_tokens"] = json!(output_length);
        } else if let Some(Some(cap)) = MAX_TOKENS_CAP.get() {
            body["max_tokens"] = json!(cap);
        }

        body.to_string()
    }

    fn extra_headers(_body: &str) -> Vec<(&'static str, String)> {
        vec![("routing-strategy", AIBRIX_ROUTE_STRATEGY.get().unwrap().clone())]
    }

    async fn parse_response(
        response: Response,
        _stream: bool,
        _timeout_duration: Duration,
        _in_flight: Option<Arc<InFlightState>>,
    ) -> Result<BTreeMap<String, String>, RequestError> {
        let mut result = BTreeMap::new();

        result.insert("status".to_string(), response.status().as_str().to_string());

        Ok(result)
    }
}
