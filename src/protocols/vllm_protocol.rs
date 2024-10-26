use std::{collections::BTreeMap, future::Future};

use rand::{thread_rng, Rng};
use reqwest::Response;
use tokenizers::Tokenizer;

use super::Protocol;

pub struct VllmProtocol {
    tokenizer: Tokenizer,

    /// Start of the token id range.
    start: u32,

    /// End of the token id range.
    end: u32,
}

impl VllmProtocol {
    /// Current the randomly generated token ids are in the range of 0..10000.
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            start: 0,
            end: 10000,
        }
    }
}

impl Protocol for VllmProtocol {
    fn request_json_body(&self, input_token_length: u64, output_token_length: u64) -> String {
        let input_token_ids = (0..input_token_length)
            .map(|_| thread_rng().gen_range(self.start..self.end))
            .collect::<Vec<_>>();
        let _input = self
            .tokenizer
            .decode(input_token_ids.as_slice(), false)
            .unwrap();
        let json_body =
            serde_json::json!({"max_tokens": output_token_length, "tokens": input_token_ids});
        json_body.to_string()
    }

    fn parse_response(response: Response, _input_token_length:Option<u64>) -> BTreeMap<String, String> {
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

            let inference_time = response
                .headers()
                .get("x-inference-time")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            map.insert("inference_time".to_string(), inference_time);

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

    fn parse_response_async(_: Response) -> impl Future<Output = BTreeMap<String, String>> {
        async { unimplemented!() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        if std::path::Path::new("/nvme/huggingface/hub/opt-1.3b/tokenizer.json").exists() {
            let tokenizer =
                Tokenizer::from_file("/nvme/huggingface/hub/opt-1.3b/tokenizer.json").unwrap();
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
