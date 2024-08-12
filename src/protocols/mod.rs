use std::collections::BTreeMap;

use reqwest::Response;

pub mod tgi_protocol;

pub mod distserve_protocol;

pub trait Protocol {
    fn request_json_body(&self, input_token_length: u64, output_token_length: u64) -> String;
    fn parse_response(response: Response) -> BTreeMap<String, String>;
}
