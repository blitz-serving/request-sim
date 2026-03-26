use reqwest::Response;
use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::dataset::PromptPayload;

pub static MODEL_NAME: OnceLock<String> = OnceLock::new();
pub static METRIC_PERCENTILES: OnceLock<Vec<u32>> = OnceLock::new();
pub static MAX_TOKENS_CAP: OnceLock<Option<u64>> = OnceLock::new();
pub static RID_SOURCE: OnceLock<RidSource> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidSource {
    None,
    ContentHash,
}

/// Compute a deterministic rid from the serialized messages JSON.
/// Returns a 16-char hex string (first 8 bytes of SHA-256).
pub fn compute_content_hash_rid(messages: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    let serialized = messages.to_string();
    let hash = Sha256::digest(serialized.as_bytes());
    hash[..8].iter().map(|b| format!("{b:02x}")).collect()
}

pub mod aibrix_api;
pub mod openai_api;
pub mod sgl_api;
pub mod tgi_api;

pub use aibrix_api::{AbxApi, AIBRIX_ROUTE_STRATEGY};
pub use openai_api::OaiApi;
pub use sgl_api::SglApi;
pub use tgi_api::TgiApi;

use std::time::Duration;

pub enum RequestError {
    Timeout,
    StreamErr(std::io::Error),
    Other(reqwest::Error),
}

#[async_trait::async_trait]
pub trait LLMApi: Copy + Clone + Send + Sync {
    const AIBRIX_PRIVATE_HEADER: bool;
    fn request_json_body(prompt: PromptPayload, output_length: u64, stream: bool) -> String;
    async fn parse_response(
        response: Response,
        stream: bool,
        timeout_duration: Duration,
    ) -> Result<BTreeMap<String, String>, RequestError>;
}
