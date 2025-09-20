use std::collections::BTreeMap;

use reqwest::Response;

pub mod tgi_api;

pub trait LLMApi: Copy {
    fn request_json_body(&self, prompt: String, output_length: u64) -> String;
    fn parse_response(response: Response) -> BTreeMap<String, String>;
}
