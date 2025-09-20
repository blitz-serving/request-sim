use std::collections::BTreeMap;

use reqwest::Response;

use super::LLMApi;

pub struct TGIApi;

impl Copy for TGIApi {}

impl Clone for TGIApi {
    fn clone(&self) -> Self {
        *self
    }
}

impl LLMApi for TGIApi {
    fn request_json_body(&self, prompt: String, output_length: u64) -> String {
        let json_body =
            serde_json::json!({"input":prompt,"parameter":{"max_new_tokens":output_length}});
        json_body.to_string()
    }

    fn parse_response(response: Response) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        map.insert("status".to_string(), response.status().as_str().to_string());
        if response.status().is_success() {
            let first_token_time = response
                .headers()
                .get("x-first-token-time")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert("first_token_time".to_string(), first_token_time);

            let total_time = response
                .headers()
                .get("x-total-time")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert("total_time".to_string(), total_time);

            let inference_time = response
                .headers()
                .get("x-inference-time")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert("inference_time".to_string(), inference_time);

            let queue_time = response
                .headers()
                .get("x-queue-time")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert("queue_time".to_string(), queue_time);

            let max_time_between_tokens = response
                .headers()
                .get("x-max-time-between-tokens")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert(
                "max_time_between_tokens".to_string(),
                max_time_between_tokens,
            );
        }
        map
    }
}
