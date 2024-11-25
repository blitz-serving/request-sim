use std::collections::BTreeMap;

use reqwest::Response;
use tokenizers::Tokenizer;

use super::Protocol;

pub struct StProtocol {
    tokenizer: Tokenizer,

    target_token: u32,
}

impl StProtocol {
    /// Current the randomly generated token ids are in the range of 0..10000.
    pub fn new(tokenizer: Tokenizer) -> Self {
        let target_token = tokenizer.encode("Hello World", false).unwrap().get_ids()[1];
        Self {
            tokenizer,
            target_token,
        }
    }
}

impl Protocol for StProtocol {
    fn request_json_body(&self, input_token_length: u64, output_token_length: u64) -> String {
        let input = vec![self.target_token; input_token_length as usize];
        let input = self.tokenizer.decode(&input, false).unwrap();
        let json_body =
            serde_json::json!({"inputs":input,"parameters":{"max_new_tokens":output_token_length}});
        json_body.to_string()
    }

    fn parse_response(&self) -> fn(response: Response) -> BTreeMap<String, String> {
        |response: Response| -> BTreeMap<String, String> {
            let mut map = BTreeMap::new();
            map.insert("status".to_string(), response.status().as_str().to_string());
            if response.status().is_success() {
                let request_id = response
                    .headers()
                    .get("x-request-id")
                    .map_or("0".to_string(), |hv| hv.to_str().unwrap().to_string());
                map.insert("request_id".to_string(), request_id);

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

                let avg_time_between_tokens = response
                    .headers()
                    .get("x-avg-time-between-tokens")
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                map.insert(
                    "avg_time_between_tokens".to_string(),
                    avg_time_between_tokens,
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

                let p95_time_between_tokens = response
                    .headers()
                    .get("x-p95-time-between-tokens")
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                map.insert(
                    "p95_time_between_tokens".to_string(),
                    p95_time_between_tokens,
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
            }
            map
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER_PATH: &str = "/nvme/huggingface/hub/opt-1.3b/tokenizer.json";
    // const TOKENIZER_PATH: &str = "/nvme/huggingface/hub/Llama-2-7b-hf/tokenizer.json";

    #[test]
    fn test_tokenizer() {
        if std::path::Path::new(TOKENIZER_PATH).exists() {
            let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
            let vocab_size = tokenizer.get_vocab_size(false);
            println!("{:?}", vocab_size);

            let encodings = tokenizer
                .encode("Hello World", false)
                .unwrap()
                .get_ids()
                .to_vec();
            println!("{:?}", encodings);

            let sentence = tokenizer.decode(&encodings, false).unwrap();
            println!("{:?}", sentence);

            let words = encodings
                .iter()
                .map(|a| tokenizer.id_to_token(*a).unwrap())
                .collect::<Vec<_>>();
            println!("{:?}", words);

            let input = vec![encodings[1]; 100 as usize];
            let input = tokenizer.decode(&input, false).unwrap();
            println!("{:?}", input);
        } else {
            print!("Tokenizer file not found");
        }
    }

    #[tokio::test]
    async fn test_st_protocol() {
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let st_protocol = StProtocol::new(tokenizer);
        let json_body = st_protocol.request_json_body(100, 100);
        println!("{}", json_body);
    }
}
