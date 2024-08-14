use std::{collections::BTreeMap, future::Future};

use super::Protocol;

pub struct MockProtocol;

impl Protocol for MockProtocol {
    fn request_json_body(&self, input_token_length: u64, output_token_length: u64) -> String {
        serde_json::json!({
            "input_token_length": input_token_length,
            "output_token_length": output_token_length,
        })
        .to_string()
    }

    fn parse_response(_: reqwest::Response) -> BTreeMap<String, String> {
        Default::default()
    }

    fn parse_response_async(
        _: reqwest::Response,
    ) -> impl Future<Output = BTreeMap<String, String>> {
        async { Default::default() }
    }
}
