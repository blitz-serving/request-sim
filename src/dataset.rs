use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::{AtomicUsize, Ordering},
};
#[cfg(feature = "prompt-text-hashed")]
use std::{cell::UnsafeCell, collections::HashMap};

#[cfg(feature = "prompt-text-hashed")]
use crate::token_sampler::TokenSampler;
#[cfg(feature = "prompt-text-hashed")]
use crate::SpinRwLock;
use chrono::NaiveDateTime;
use request_sim_macros::prompt_text;
use serde::{Deserialize, Serialize};
#[cfg(feature = "prompt-text-hashed")]
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
#[cfg(feature = "prompt-text-hashed")]
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
#[cfg(feature = "prompt-text-hashed")]
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

    #[cfg(feature = "prompt-text-hashed")]
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

#[cfg(feature = "prompt-text-hashed")]
impl BailianDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(HashMap::new()),
            rwlock: SpinRwLock::new(),
        }
    }
}

#[cfg(feature = "prompt-text-hashed")]
unsafe impl Send for BailianDataset {}
#[cfg(feature = "prompt-text-hashed")]
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

#[cfg(feature = "prompt-text-hashed")]
impl MooncakeDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(HashMap::new()),
            rwlock: SpinRwLock::new(),
        }
    }
}

#[cfg(feature = "prompt-text-hashed")]
unsafe impl Send for MooncakeDataset {}
#[cfg(feature = "prompt-text-hashed")]
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
// ============== OpenAIDataset ==============
//

/// Raw JSONL line from OpenAI Chat Completions format.
/// Extra fields (model, stream, temperature, tools, metadata, etc.) are ignored.
#[cfg(feature = "prompt-text-plain")]
#[derive(Debug, Deserialize)]
struct OpenaiRawItem {
    messages: serde_json::Value,
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// OpenAI dataset stores the actual messages and sends them directly.
/// Unlike mooncake/bailian (which only have token counts and need TokenSampler
/// to generate synthetic text), OpenAI format already contains the real prompt.
#[cfg(feature = "prompt-text-plain")]
struct OpenaiItem {
    messages: serde_json::Value,
    input_length: u64,
    output_length: u64,
}

#[prompt_text(plain)]
pub struct OpenaiDataset {
    items: Vec<OpenaiItem>,
}

#[cfg(feature = "prompt-text-plain")]
impl OpenaiDataset {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
}

#[prompt_text(plain)]
impl LLMTrace for OpenaiDataset {
    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        for (lineno, line) in BufReader::new(file).lines().enumerate() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            let raw: OpenaiRawItem = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("Failed to parse OpenAI JSONL line {}: {e}", lineno + 1));
            self.items.push(OpenaiItem {
                messages: raw.messages,
                input_length: raw.prompt_tokens,
                output_length: raw.completion_tokens,
            });
        }
        tracing::info!("Loaded OpenAI dataset: {} items", self.items.len());
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> DataIter {
        DataIter { size: self.items.len(), index: AtomicUsize::new(0) }
    }

    fn rps(&self) -> f64 {
        1.0
    }

    fn timestamp(&self, index: usize) -> u64 {
        index as u64
    }

    fn inflate(&self, index: usize) -> (PromptPayload, u64, u64) {
        let item = &self.items[index];
        (
            PromptPayload::Messages(item.messages.clone()),
            item.input_length,
            item.output_length,
        )
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
