use std::collections::BTreeMap;

use reqwest::Response;

pub mod distserve_api;
pub mod tgi_api;

pub use distserve_api::DistserveApi;
pub use tgi_api::TGIApi;

pub trait LLMApi: Copy + Clone {
    fn request_json_body(prompt: String, output_length: u64) -> String;
    fn parse_response(response: Response) -> BTreeMap<String, String>;
}
