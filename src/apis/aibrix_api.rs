use reqwest::Response;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

use super::{LLMApi, RequestError, MAX_TOKENS_CAP, MODEL_NAME};

#[derive(Copy, Clone)]
pub struct AbxApi;

pub static AIBRIX_ROUTE_STRATEGY: OnceLock<String> = OnceLock::new();

#[async_trait::async_trait]
impl LLMApi for AbxApi {
    const AIBRIX_PRIVATE_HEADER: bool = true;

    fn request_json_body(prompt: String, output_length: u64, stream: bool) -> String {
        let mut body = json!({
            "model": MODEL_NAME.get().unwrap().as_str(),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": stream,
        });
        if output_length > 0 {
            body["min_tokens"] = json!(output_length);
            body["max_tokens"] = json!(output_length);
        } else if let Some(Some(cap)) = MAX_TOKENS_CAP.get() {
            body["max_tokens"] = json!(cap);
        }

        body.to_string()
    }

    async fn parse_response(
        response: Response,
        _stream: bool,
        _timeout_duration: Duration,
    ) -> Result<BTreeMap<String, String>, RequestError> {
        let mut result = BTreeMap::new();

        result.insert("status".to_string(), response.status().as_str().to_string());

        Ok(result)
    }
}
