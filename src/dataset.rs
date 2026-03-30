use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::{AtomicUsize, Ordering},
};
#[cfg(not(feature = "prompt-text-plain"))]
use std::{cell::UnsafeCell, collections::HashMap};

#[cfg(not(feature = "prompt-text-plain"))]
use crate::token_sampler::TokenSampler;
#[cfg(not(feature = "prompt-text-plain"))]
use crate::SpinRwLock;
use chrono::NaiveDateTime;
use request_sim_macros::prompt_text;
use serde::{Deserialize, Serialize};
#[cfg(feature = "prompt-text-plain")]
use serde_json::json;
#[cfg(not(feature = "prompt-text-plain"))]
use tracing::{instrument, Level};

/// Describes the format of the prompt returned by `inflate()`.
///
/// - `Content`: raw text to be wrapped in `[{"role":"user","content":...}]` by the API layer.
/// - `Messages`: pre-structured messages array, used directly as the `"messages"` value.
#[derive(Clone, Debug)]
pub enum PromptPayload {
    Content(String),
    Messages(serde_json::Value),
}

/// jsonl of Bailian
#[cfg(not(feature = "prompt-text-plain"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BailianDataItem {
    pub chat_id: i64,
    pub parent_chat_id: i64,
    pub timestamp: f64,
    pub input_length: u64,
    pub output_length: u64,
    pub r#type: String,
    pub turn: u64,
    pub hash_ids: Vec<u64>,
}

/// jsonl of Mooncake
#[cfg(not(feature = "prompt-text-plain"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MooncakeDataItem {
    pub timestamp: f32,
    pub input_length: u64,
    pub output_length: u64,
    pub hash_ids: Vec<u64>,
}

#[allow(dead_code)]
fn from_timestamp<'de, D>(deserializer: D) -> Result<NaiveDateTime, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    // support "2023-11-16 18:15:46.6805900" format
    NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S%.f").map_err(serde::de::Error::custom)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AzureDataItem {
    #[serde(rename = "TIMESTAMP", deserialize_with = "from_timestamp")]
    pub naive_timestamp: NaiveDateTime,
    #[serde(rename = "ContextTokens")]
    pub context_tokens: u64,
    #[serde(rename = "GeneratedTokens")]
    pub generated_tokens: u64,
    #[serde(skip)]
    pub timestamp: u64,
}

pub struct DataIter {
    size: usize,
    index: AtomicUsize,
}

impl Iterator for DataIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index.fetch_add(1, Ordering::AcqRel);
        if i >= self.size {
            // fuse the iterator
            self.index.store(i, Ordering::Release);
            return None;
        }
        Some(i)
    }
}

unsafe impl Send for DataIter {}
unsafe impl Sync for DataIter {}

pub trait LLMTrace: Send + Sync {
    fn load(&mut self, path: &str);
    fn len(&self) -> usize;
    fn timestamp(&self, index: usize) -> u64;

    #[cfg(not(feature = "prompt-text-plain"))]
    fn inflate(&self, index: usize, ts: &TokenSampler) -> (PromptPayload, u64, u64);

    #[cfg(feature = "prompt-text-plain")]
    fn inflate(&self, index: usize) -> (PromptPayload, u64, u64);

    fn iter(&self) -> DataIter;
    fn rps(&self) -> f64;
}

//
// ============== BailianDataset ==============
//
#[prompt_text(hashed)]
pub struct BailianDataset {
    items: Vec<BailianDataItem>,
    user_prompts: UnsafeCell<HashMap<u64, String>>,
    rwlock: SpinRwLock,
}

#[cfg(not(feature = "prompt-text-plain"))]
impl BailianDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(HashMap::new()),
            rwlock: SpinRwLock::new(),
        }
    }
}

#[cfg(not(feature = "prompt-text-plain"))]
unsafe impl Send for BailianDataset {}
#[cfg(not(feature = "prompt-text-plain"))]
unsafe impl Sync for BailianDataset {}

#[prompt_text(hashed)]
impl LLMTrace for BailianDataset {
    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();

        for line in BufReader::new(file).lines() {
            let item: BailianDataItem = serde_json::from_str(line.unwrap().as_str()).unwrap();
            self.items.push(item);
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> DataIter {
        DataIter { size: self.items.len(), index: AtomicUsize::new(0) }
    }

    fn rps(&self) -> f64 {
        self.items.len() as f64
            / (self.items.last().unwrap().timestamp - self.items.first().unwrap().timestamp)
    }

    fn timestamp(&self, index: usize) -> u64 {
        (self.items[index].timestamp * 1000.) as u64
    }

    #[instrument(skip_all, target = "inflate", fields(chat_id = index), level = Level::INFO)]
    fn inflate(&self, index: usize, ts: &TokenSampler) -> (PromptPayload, u64, u64) {
        // NOTE: the last block hash may be hashed onto a partially filled block
        const BLOCK_SIZE: usize = 16;
        unsafe {
            let data_item = self.items.get(index).unwrap();
            let last_block_len =
                (*data_item).input_length as usize - ((*data_item).hash_ids.len() - 1) * BLOCK_SIZE;
            debug_assert!(last_block_len <= BLOCK_SIZE);

            let x = if last_block_len == BLOCK_SIZE { 0 } else { 1 };
            let mut prompt =
                String::with_capacity(usize::next_power_of_two((*data_item).input_length as usize));
            for &hash_id in (*data_item).hash_ids.iter().take((*data_item).hash_ids.len() - x) {
                // loop invariant: rwlock is free
                self.rwlock.read_lock();
                if let Some(s) = (&*self.user_prompts.get()).get(&hash_id) {
                    prompt.push_str(&s);
                    self.rwlock.read_unlock();
                } else {
                    self.rwlock.read_unlock();
                    let s = ts.gen_string(BLOCK_SIZE);
                    self.rwlock.write_lock();
                    if let Some(s0) = (*self.user_prompts.get()).get(&hash_id) {
                        prompt.push_str(&s0);
                    } else {
                        prompt.push_str(&s);
                        (&mut *self.user_prompts.get()).insert(hash_id, s);
                    }
                    self.rwlock.write_unlock();
                }
            }

            if x == 1 {
                let last_block_prompt = ts.gen_string(last_block_len);
                prompt.push_str(&last_block_prompt);
                self.rwlock.write_lock();
                (&mut *self.user_prompts.get())
                    .insert(*(*data_item).hash_ids.last().unwrap(), last_block_prompt);
                self.rwlock.write_unlock();
            }

            (PromptPayload::Content(prompt), (*data_item).input_length, (*data_item).output_length)
        }
    }
}

//
// ============== MooncakeDataset ==============
//
#[prompt_text(hashed)]
pub struct MooncakeDataset {
    items: Vec<MooncakeDataItem>,
    user_prompts: UnsafeCell<HashMap<u64, String>>,
    rwlock: SpinRwLock,
}

#[cfg(not(feature = "prompt-text-plain"))]
impl MooncakeDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(HashMap::new()),
            rwlock: SpinRwLock::new(),
        }
    }
}

#[cfg(not(feature = "prompt-text-plain"))]
unsafe impl Send for MooncakeDataset {}
#[cfg(not(feature = "prompt-text-plain"))]
unsafe impl Sync for MooncakeDataset {}

#[prompt_text(hashed)]
impl LLMTrace for MooncakeDataset {
    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        for line in BufReader::new(file).lines() {
            let item: MooncakeDataItem = serde_json::from_str(line.unwrap().as_str()).unwrap();
            self.items.push(item);
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> DataIter {
        DataIter { size: self.items.len(), index: AtomicUsize::new(0) }
    }

    fn rps(&self) -> f64 {
        self.items.len() as f64
            / (self.items.last().unwrap().timestamp as f64
                - self.items.first().unwrap().timestamp as f64)
    }

    fn timestamp(&self, index: usize) -> u64 {
        (self.items[index].timestamp * 1000.) as u64
    }

    fn inflate(&self, index: usize, ts: &TokenSampler) -> (PromptPayload, u64, u64) {
        // NOTE: the last block hash may be hashed onto a partially filled block
        const BLOCK_SIZE: usize = 512;
        unsafe {
            let data_item = self.items.get(index).unwrap();
            let last_block_len =
                (*data_item).input_length as usize - ((*data_item).hash_ids.len() - 1) * BLOCK_SIZE;
            debug_assert!(last_block_len <= BLOCK_SIZE);

            let x = if last_block_len == BLOCK_SIZE { 0 } else { 1 };
            let mut prompt =
                String::with_capacity(usize::next_power_of_two((*data_item).input_length as usize));
            for &hash_id in (*data_item).hash_ids.iter().take((*data_item).hash_ids.len() - x) {
                // loop invariant: rwlock is free
                self.rwlock.read_lock();
                if let Some(s) = (&*self.user_prompts.get()).get(&hash_id) {
                    prompt.push_str(&s);
                    self.rwlock.read_unlock();
                } else {
                    self.rwlock.read_unlock();
                    let s = ts.gen_string(BLOCK_SIZE);
                    self.rwlock.write_lock();
                    if let Some(s0) = (*self.user_prompts.get()).get(&hash_id) {
                        prompt.push_str(&s0);
                    } else {
                        prompt.push_str(&s);
                        (&mut *self.user_prompts.get()).insert(hash_id, s);
                    }
                    self.rwlock.write_unlock();
                }
            }
            // postcond: rwlock is free

            if x == 1 {
                let last_block_prompt = ts.gen_string(last_block_len);
                prompt.push_str(&last_block_prompt);
                self.rwlock.write_lock();
                (&mut *self.user_prompts.get())
                    .insert(*(*data_item).hash_ids.last().unwrap(), last_block_prompt);
                self.rwlock.write_unlock();
            }

            (PromptPayload::Content(prompt), (*data_item).input_length, (*data_item).output_length)
        }
    }
}

//
// ============== MiniMaxDataset ==============
//

/// Raw JSONL line from MiniMax traces.
#[cfg(feature = "prompt-text-plain")]
#[derive(Debug, Deserialize)]
struct MiniMaxRawItem {
    server_timestamp: u64,
    dialogue_input: String,
    dialogue_outputs: Option<String>,
}

/// A single message entry inside `dialogue_input.data[]`.
#[cfg(feature = "prompt-text-plain")]
#[derive(Debug, Deserialize)]
struct MiniMaxDialogueMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    text: String,
}

/// Wrapper for `dialogue_input` JSON (we only need the `data` array).
#[cfg(feature = "prompt-text-plain")]
#[derive(Debug, Deserialize)]
struct MiniMaxDialogueInput {
    data: Vec<MiniMaxDialogueMessage>,
}

/// A single output entry inside the `dialogue_outputs` JSON array.
#[cfg(feature = "prompt-text-plain")]
#[derive(Debug, Deserialize)]
struct MiniMaxOutputEntry {
    model_input_tokens_count: u64,
    output_tokens_count: Vec<u64>,
}

/// Processed item ready for replay.
#[cfg(feature = "prompt-text-plain")]
struct MiniMaxParsedItem {
    timestamp_ms: u64,
    prompt: PromptPayload,
    input_length: u64,
    output_length: u64,
}

#[prompt_text(plain)]
pub struct MiniMaxDataset {
    items: Vec<MiniMaxParsedItem>,
}

#[cfg(feature = "prompt-text-plain")]
impl MiniMaxDataset {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
}

#[cfg(feature = "prompt-text-plain")]
unsafe impl Send for MiniMaxDataset {}
#[cfg(feature = "prompt-text-plain")]
unsafe impl Sync for MiniMaxDataset {}

#[prompt_text(plain)]
impl LLMTrace for MiniMaxDataset {
    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        let mut raw_items: Vec<MiniMaxRawItem> = Vec::new();

        for line in BufReader::new(file).lines() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            let item: MiniMaxRawItem = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("Failed to parse MiniMax JSONL line: {e}"));
            raw_items.push(item);
        }

        // Sort by timestamp
        raw_items.sort_by_key(|item| item.server_timestamp);

        let base_ts = raw_items.first().map(|i| i.server_timestamp).unwrap_or(0);

        for raw in &raw_items {
            // Skip entries with null dialogue_outputs
            let dialogue_outputs = match &raw.dialogue_outputs {
                Some(s) => s,
                None => continue,
            };
            // Parse dialogue_outputs: JSON array of output entries
            let outputs: Vec<MiniMaxOutputEntry> =
                serde_json::from_str(dialogue_outputs)
                    .unwrap_or_else(|e| panic!("Failed to parse dialogue_outputs: {e}"));
            let first_output = outputs.first().expect("dialogue_outputs array is empty");
            let input_length = first_output.model_input_tokens_count;
            let output_length = *first_output
                .output_tokens_count
                .first()
                .expect("output_tokens_count array is empty");

            // Parse dialogue_input: JSON object with `data` array of messages
            let dialogue: MiniMaxDialogueInput =
                serde_json::from_str(&raw.dialogue_input)
                    .unwrap_or_else(|e| panic!("Failed to parse dialogue_input: {e}"));

            // Build structured messages array, mapping MiniMax roles to OpenAI roles
            let messages: Vec<serde_json::Value> = dialogue
                .data
                .iter()
                .map(|msg| {
                    let role = match msg.role.as_str() {
                        "" => "user",
                        "ai" => "assistant",
                        other => other,
                    };
                    json!({"role": role, "content": msg.text})
                })
                .collect();

            // Convert nanoseconds to relative milliseconds
            let timestamp_ms = (raw.server_timestamp - base_ts) / 1_000_000;

            self.items.push(MiniMaxParsedItem {
                timestamp_ms,
                prompt: PromptPayload::Messages(json!(messages)),
                input_length,
                output_length,
            });
        }

        tracing::info!(
            "Loaded MiniMax dataset: {} items, time span {:.1}s",
            self.items.len(),
            self.items.last().map(|i| i.timestamp_ms as f64 / 1000.0).unwrap_or(0.0),
        );
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> DataIter {
        DataIter {
            size: self.items.len(),
            index: AtomicUsize::new(0),
        }
    }

    fn rps(&self) -> f64 {
        if self.items.len() < 2 {
            return 1.0;
        }
        let duration_s = (self.items.last().unwrap().timestamp_ms
            - self.items.first().unwrap().timestamp_ms) as f64
            / 1000.0;
        if duration_s <= 0.0 {
            return 1.0;
        }
        self.items.len() as f64 / duration_s
    }

    fn timestamp(&self, index: usize) -> u64 {
        self.items[index].timestamp_ms
    }

    fn inflate(&self, index: usize) -> (PromptPayload, u64, u64) {
        let item = &self.items[index];
        (item.prompt.clone(), item.input_length, item.output_length)
    }
}

//
// ============== PlainTextDataset ==============
//

#[prompt_text(plain)]
pub struct PlainTextDataset {
    items: Vec<String>,
}

#[cfg(feature = "prompt-text-plain")]
impl PlainTextDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }
}

#[cfg(feature = "prompt-text-plain")]
unsafe impl Send for PlainTextDataset {}
#[cfg(feature = "prompt-text-plain")]
unsafe impl Sync for PlainTextDataset {}

#[prompt_text(plain)]
impl LLMTrace for PlainTextDataset {
    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        for line in BufReader::new(file).lines() {
            let line = line.unwrap();
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            self.items.push(trimmed.to_string());
        }
        tracing::info!(
            "Loaded PlainTextDataset: {} items",
            self.items.len(),
        );
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> DataIter {
        DataIter {
            size: self.items.len(),
            index: AtomicUsize::new(0),
        }
    }

    fn rps(&self) -> f64 {
        1.0
    }

    fn timestamp(&self, index: usize) -> u64 {
        index as u64
    }

    fn inflate(&self, index: usize) -> (PromptPayload, u64, u64) {
        (PromptPayload::Content(self.items[index].clone()), 0, 0)
    }
}
