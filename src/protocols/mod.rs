use std::{collections::BTreeMap, future::Future};

use reqwest::Response;

pub mod distserve_protocol;
pub mod mock_protocol;
pub mod st_protocol;
pub mod vllm_protocol;

pub use distserve_protocol::DistserveProtocol;
pub use mock_protocol::MockProtocol;
pub use st_protocol::StProtocol;
pub use vllm_protocol::VllmProtocol;

pub trait Protocol {
    fn request_json_body(&self, input_token_length: u64, output_token_length: u64) -> String;
    fn parse_response(response: Response, input_length:Option<u64>) -> BTreeMap<String, String>;

    /// New method to parse response asynchronously with generic output type.
    fn parse_response_async(response: Response) -> impl Future<Output = BTreeMap<String, String>>;
}
