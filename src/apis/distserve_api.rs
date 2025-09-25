use std::collections::BTreeMap;

use reqwest::Response;

use super::LLMApi;

pub struct DistserveApi {}

impl Copy for DistserveApi {}

impl Clone for DistserveApi {
    fn clone(&self) -> Self {
        *self
    }
}

impl LLMApi for DistserveApi {
    fn request_json_body(&self, prompt: String, output_length: u64) -> String {
        unimplemented!("this api maybe out dated, checkout for a new one!");
        let json_body = serde_json::json!({
            "prompt": prompt,
            "max_tokens": output_length,
            "n":1,
            "best_of":1,
            "use_beam_search": false,
            "temperature": 1.0,
            "top_p": 1.0,
            "ignore_eos": true,
            "stream": false
        });
        json_body.to_string()
    }

    fn parse_response(&self, response: Response) -> BTreeMap<String, String> {
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

            let p70_time_between_tokens = response
                .headers()
                .get("x-p70-time-between-tokens")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert(
                "p70_time_between_tokens".to_string(),
                p70_time_between_tokens,
            );

            let p90_time_between_tokens = response
                .headers()
                .get("x-p90-time-between-tokens")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert(
                "p90_time_between_tokens".to_string(),
                p90_time_between_tokens,
            );

            let p99_time_between_tokens = response
                .headers()
                .get("x-p99-time-between-tokens")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert(
                "p99_time_between_tokens".to_string(),
                p99_time_between_tokens,
            );

            let output_length = response
                .headers()
                .get("x-output-length")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert("output_length".to_string(), output_length);

            let input_length = response
                .headers()
                .get("x-input-length")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert("input_length".to_string(), input_length);
        }
        map
    }
}
