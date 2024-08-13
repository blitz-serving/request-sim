use std::collections::BTreeMap;

use reqwest::Response;

pub mod tgi_protocol;
pub mod vllm_protocol;

pub trait Protocol {
    type Parsed;

    fn request_json_body(&self, input_token_length: u64, output_token_length: u64) -> String;
    fn parse_response(response: Response) -> BTreeMap<String, String>;

    /// New method to parse response asynchronously with generic output type.
    fn parse_response_async(response: Response) -> impl std::future::Future<Output = Self::Parsed>;
}
