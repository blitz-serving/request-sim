use std::collections::BTreeMap;

use rand::{thread_rng, Rng};
use reqwest::Response;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::Protocol;

pub struct DistserveProtocol {
    tokenizer: Tokenizer,

    /// Start of the token id range.
    start: u32,

    /// End of the token id range.
    end: u32,

    max_token_size: u64,
}

impl DistserveProtocol {
    /// Current the randomly generated token ids are in the range of 0..10000.
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            start: 0,
            end: 10000,
            max_token_size: 3950,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct LifetimeEvent {
    timestamp: f64,
    event_type: String,
}
#[derive(Serialize, Deserialize, Debug)]
struct DistserveResponse {
    text: String,
    timestamps: Vec<f64>,
    lifetime_events: Vec<LifetimeEvent>,
}

impl Protocol for DistserveProtocol {
    fn request_json_body(&self, input_token_length: u64, output_token_length: u64) -> String {
        let truncated_input_length;
        let truncated_output_length;
        if input_token_length + output_token_length >= self.max_token_size {
            truncated_input_length = 3900;
            truncated_output_length = 49;
            //println!("trucated length {} {}", truncated_input_length, truncated_output_length);
        } else {
            truncated_input_length = input_token_length;
            truncated_output_length = output_token_length;
            //println!("original length {} {}", truncated_input_length, truncated_output_length);
        }
        let input_token_ids = (0..truncated_input_length)
            .map(|_| thread_rng().gen_range(self.start..self.end))
            .collect::<Vec<_>>();
        //println!("vector length: {}", input_token_ids.len());
        let _input = self
            .tokenizer
            .decode(input_token_ids.as_slice(), false)
            .unwrap();
        //println!("prmopt: {}", input);
        let json_body = serde_json::json!({
            //"prompt":input,
            "prompt_token_ids": input_token_ids,
            "max_tokens":truncated_output_length,
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

    fn parse_response(response: Response) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        println!("{:?}", response);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        if std::path::Path::new("/nvme/huggingface/hub/Llama-2-7b-hf/tokenizer.json").exists() {
            let tokenizer =
                Tokenizer::from_file("/nvme/huggingface/hub/Llama-2-7b-hf/tokenizer.json").unwrap();
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
            let input_token_ids = (0..100)
                .map(|_| thread_rng().gen_range(0..(tokenizer.get_vocab_size(false) as u32)))
                .collect::<Vec<_>>();
            let json_body = serde_json::json!({"input":tokenizer.decode(&input_token_ids, false).unwrap(),"parameter":{"max_new_tokens":100}});
            println!("{}", json_body.to_string());
        } else {
            print!("Tokenizer file not found");
        }
    }
}
