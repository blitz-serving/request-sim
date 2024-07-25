use std::collections::BTreeMap;

use rand::{thread_rng, Rng};
use reqwest::Response;
use tokenizers::Tokenizer;

use super::Protocol;

pub struct TgiProtocol {
    tokenizer: Tokenizer,
    start: u32,
    end: u32,
}

impl Protocol for TgiProtocol {
    fn request_json_body(&self, input_token_length: u64, output_token_length: u64) -> String {
        let input_token_ids = (0..input_token_length)
            .map(|_| thread_rng().gen_range(self.start..self.end))
            .collect::<Vec<_>>();
        let input = self
            .tokenizer
            .decode(input_token_ids.as_slice(), false)
            .unwrap();
        let json_body =
            serde_json::json!({"input":input,"parameter":{"max_new_tokens":output_token_length}});
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let tokenizer =
            Tokenizer::from_file("/nvme/huggingface/hub/Llama-2-7b-hf/tokenizer.json").unwrap();

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

        let input_token_ids = (0..100)
            .map(|_| thread_rng().gen_range(0..(tokenizer.get_vocab_size(false) as u32)))
            .collect::<Vec<_>>();

        let json_body = serde_json::json!({"input":tokenizer.decode(&input_token_ids, false).unwrap(),"parameter":{"max_new_tokens":100}});
        println!("{}", json_body.to_string());
    }
}
