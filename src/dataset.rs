use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::{AtomicUsize, Ordering},
};
#[cfg(feature = "prompt-text-hashed")]
use std::{cell::UnsafeCell, collections::HashMap, collections::HashSet};

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
/// - `Body`: pre-structured request body for backends with proprietary formats (e.g. Amadeus).
#[derive(Clone, Debug)]
pub enum PromptPayload {
    Content(String),
    Messages(serde_json::Value),
    Body(serde_json::Value),
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

    /// Build a ConversationGraph from the loaded dataset.
    /// Returns None for datasets that don't support multi-turn tracking.
    #[cfg(feature = "prompt-text-hashed")]
    fn build_conversation_graph(&self) -> Option<ConversationGraph> {
        None
    }

    /// Inflate a multi-turn entry as a Messages array with proper role structure.
    /// `ancestor_chain`: ordered data_indices from root to parent [root, ..., parent].
    /// `ancestor_outputs`: captured output text for each ancestor, same order.
    /// Returns None if the dataset doesn't support multi-turn Messages.
    #[cfg(feature = "prompt-text-hashed")]
    fn inflate_as_messages(
        &self,
        _index: usize,
        _ts: &TokenSampler,
        _graph: &ConversationGraph,
        _ancestor_chain: &[usize],
        _ancestor_outputs: &[String],
    ) -> Option<(PromptPayload, u64, u64)> {
        None
    }
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

    /// Inflate an arbitrary slice of hash_ids into a string.
    /// Reuses `user_prompts` for congruence (same hash → same text).
    /// Each block is BOS/EOS-framed by TokenSampler, so enc distributes over app.
    pub(crate) fn inflate_hashes(
        &self,
        hash_ids: &[u64],
        total_tokens: usize,
        ts: &TokenSampler,
    ) -> String {
        const BLOCK_SIZE: usize = 16;
        if hash_ids.is_empty() {
            return String::new();
        }
        let last_block_len = total_tokens.saturating_sub((hash_ids.len() - 1) * BLOCK_SIZE);
        let last_block_len = last_block_len.min(BLOCK_SIZE).max(1);

        let x = if last_block_len == BLOCK_SIZE { 0 } else { 1 };
        let mut prompt = String::with_capacity(total_tokens * 4);

        unsafe {
            for &hash_id in hash_ids.iter().take(hash_ids.len() - x) {
                self.rwlock.read_lock();
                if let Some(s) = (&*self.user_prompts.get()).get(&hash_id) {
                    prompt.push_str(s);
                    self.rwlock.read_unlock();
                } else {
                    self.rwlock.read_unlock();
                    let s = ts.gen_string(BLOCK_SIZE);
                    self.rwlock.write_lock();
                    if let Some(s0) = (*self.user_prompts.get()).get(&hash_id) {
                        prompt.push_str(s0);
                    } else {
                        prompt.push_str(&s);
                        (&mut *self.user_prompts.get()).insert(hash_id, s);
                    }
                    self.rwlock.write_unlock();
                }
            }

            if x == 1 {
                let last_hash = *hash_ids.last().unwrap();
                self.rwlock.read_lock();
                if let Some(s) = (&*self.user_prompts.get()).get(&last_hash) {
                    prompt.push_str(s);
                    self.rwlock.read_unlock();
                } else {
                    self.rwlock.read_unlock();
                    let s = ts.gen_string(last_block_len);
                    self.rwlock.write_lock();
                    if let Some(s0) = (*self.user_prompts.get()).get(&last_hash) {
                        prompt.push_str(s0);
                    } else {
                        prompt.push_str(&s);
                        (&mut *self.user_prompts.get()).insert(last_hash, s);
                    }
                    self.rwlock.write_unlock();
                }
            }
        }
        prompt
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

    fn build_conversation_graph(&self) -> Option<ConversationGraph> {
        Some(ConversationGraph::build(&self.items, 16))
    }

    fn inflate_as_messages(
        &self,
        index: usize,
        ts: &TokenSampler,
        graph: &ConversationGraph,
        ancestor_chain: &[usize],
        ancestor_outputs: &[String],
    ) -> Option<(PromptPayload, u64, u64)> {
        if ancestor_chain.is_empty() {
            return None;
        }

        let item = &self.items[index];
        let mut messages = Vec::new();

        // Root turn: user message from all its hash_ids
        let root_idx = ancestor_chain[0];
        let root_item = &self.items[root_idx];
        let root_text =
            self.inflate_hashes(&root_item.hash_ids, root_item.input_length as usize, ts);
        messages.push(serde_json::json!({"role": "user", "content": root_text}));

        // Each subsequent ancestor: assistant output + user remaining delta
        for (i, &anc_idx) in ancestor_chain.iter().enumerate() {
            // Assistant message: captured output from this ancestor
            if i < ancestor_outputs.len() {
                messages.push(
                    serde_json::json!({"role": "assistant", "content": ancestor_outputs[i]}),
                );
            }

            // User message: remaining delta for the NEXT turn in the chain
            let next_idx = if i + 1 < ancestor_chain.len() {
                ancestor_chain[i + 1]
            } else {
                index // current item
            };

            let next_item = &self.items[next_idx];
            let delta = &graph.delta_hashes[next_idx];
            let output_blocks = graph.parent_output_blocks[next_idx];
            let remaining_hashes: Vec<u64> = delta
                .iter()
                .skip(output_blocks)
                .copied()
                .collect();

            if !remaining_hashes.is_empty() {
                let anc_item = &self.items[anc_idx];
                let remaining_tokens = (next_item.input_length as usize)
                    .saturating_sub(anc_item.input_length as usize)
                    .saturating_sub(anc_item.output_length as usize);
                let remaining_tokens = remaining_tokens.max(remaining_hashes.len()); // at least 1 per hash

                let remaining_text =
                    self.inflate_hashes(&remaining_hashes, remaining_tokens, ts);
                messages.push(serde_json::json!({"role": "user", "content": remaining_text}));
            }
        }

        Some((
            PromptPayload::Messages(serde_json::Value::Array(messages)),
            item.input_length,
            item.output_length,
        ))
    }
}

// ── Conversation graph for multi-turn output tracking ──────────────────────

/// Per-item metadata linking turns in a multi-turn conversation.
/// Built from `chat_id` / `parent_chat_id` after dataset load.
#[cfg(feature = "prompt-text-hashed")]
pub struct ConversationGraph {
    /// data_index of the parent turn (None for first turns or orphans).
    pub parent_index: Vec<Option<usize>>,
    /// Hash IDs in this item that do NOT appear in the parent item (ordered).
    pub delta_hashes: Vec<Vec<u64>>,
    /// Number of output blocks from the parent turn: ceil(parent.output_length / BLOCK_SIZE).
    pub parent_output_blocks: Vec<usize>,
}

#[cfg(feature = "prompt-text-hashed")]
impl ConversationGraph {
    /// Build the conversation graph from a loaded BailianDataset.
    pub(crate) fn build(items: &[BailianDataItem], block_size: usize) -> Self {
        let n = items.len();
        let mut parent_index = vec![None; n];
        let mut delta_hashes = vec![Vec::new(); n];
        let mut parent_output_blocks = vec![0usize; n];

        // Map chat_id → data_index for parent lookup
        let mut chat_to_index: HashMap<i64, usize> = HashMap::with_capacity(n);
        for (i, item) in items.iter().enumerate() {
            chat_to_index.insert(item.chat_id, i);
        }

        for (i, item) in items.iter().enumerate() {
            if let Some(&pi) = chat_to_index.get(&item.parent_chat_id) {
                // Skip self-referencing entries (turn 1 where parent_chat_id == chat_id)
                if pi == i {
                    continue;
                }
                parent_index[i] = Some(pi);

                let parent = &items[pi];
                let parent_hash_set: HashSet<u64> = parent.hash_ids.iter().copied().collect();
                delta_hashes[i] = item
                    .hash_ids
                    .iter()
                    .copied()
                    .filter(|h| !parent_hash_set.contains(h))
                    .collect();
                parent_output_blocks[i] =
                    (parent.output_length as usize + block_size - 1) / block_size;
            }
        }

        let linked = parent_index.iter().filter(|p| p.is_some()).count();
        tracing::info!(
            "ConversationGraph: {n} items, {linked} linked turns, block_size={block_size}"
        );

        Self { parent_index, delta_hashes, parent_output_blocks }
    }

    /// Returns the ordered ancestor chain for a given data_index.
    /// Returns [root, ..., parent] (not including the item itself).
    pub fn get_chain(&self, index: usize) -> Vec<usize> {
        let mut chain = Vec::new();
        let mut cur = index;
        while let Some(pi) = self.parent_index[cur] {
            chain.push(pi);
            cur = pi;
        }
        chain.reverse();
        chain
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

//
// ============== AmadeusDataset ==============
//

/// Internal storage for a single amadeus-replay JSONL line.
/// The `body` field preserves the original dict (quest_code, data, model_control, etc.)
/// so the AmadeusApi backend can send it with minimal transformation.
#[cfg(feature = "prompt-text-plain")]
struct AmadeusItem {
    body: serde_json::Value,
    input_length: u64,
    output_length: u64,
    timestamp_ms: u64,
}

#[prompt_text(plain)]
pub struct AmadeusDataset {
    items: Vec<AmadeusItem>,
}

#[cfg(feature = "prompt-text-plain")]
impl AmadeusDataset {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
}

#[prompt_text(plain)]
impl LLMTrace for AmadeusDataset {
    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        for (lineno, line) in BufReader::new(file).lines().enumerate() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            let raw: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("Failed to parse amadeus-replay JSONL line {}: {e}", lineno + 1));

            // Extract conversation data for input length estimation
            let conversation = raw.get("data").and_then(|v| v.as_array());
            let total_chars: usize = conversation
                .map(|arr| {
                    arr.iter()
                        .filter_map(|turn| turn.get("text").and_then(|t| t.as_str()))
                        .map(|s| s.len())
                        .sum()
                })
                .unwrap_or(0);
            let input_length = (total_chars / 3) as u64; // conservative char-to-token estimate

            // Extract output length from model_control.tokens_to_generate (default 256)
            let output_length = raw
                .get("model_control")
                .and_then(|mc| mc.get("tokens_to_generate"))
                .and_then(|v| v.as_u64())
                .unwrap_or(256);

            // Extract timestamp (milliseconds), support both string and integer
            let timestamp_ms = raw
                .get("timestamp")
                .map(|v| match v {
                    serde_json::Value::Number(n) => n.as_u64().unwrap_or(0),
                    serde_json::Value::String(s) => s.parse::<u64>().unwrap_or(0),
                    _ => 0,
                })
                .unwrap_or(lineno as u64); // fallback: use line index

            // Build the body to preserve for the API backend.
            // Keep: data, quest_code, model_control, system_data, functions, function_call, tools
            let mut body = serde_json::Map::new();
            if let Some(data) = raw.get("data") {
                body.insert("data".into(), data.clone());
            }
            let quest_code = raw
                .get("quest_code")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("talkie_abab55");
            body.insert("quest_code".into(), serde_json::Value::String(quest_code.to_string()));
            if let Some(mc) = raw.get("model_control") {
                body.insert("model_control".into(), mc.clone());
            }
            for key in &["system_data", "functions", "function_call", "tools"] {
                if let Some(v) = raw.get(*key) {
                    body.insert((*key).to_string(), v.clone());
                }
            }

            self.items.push(AmadeusItem {
                body: serde_json::Value::Object(body),
                input_length,
                output_length,
                timestamp_ms,
            });
        }
        tracing::info!("Loaded AmadeusDataset: {} items", self.items.len());
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> DataIter {
        DataIter { size: self.items.len(), index: AtomicUsize::new(0) }
    }

    fn rps(&self) -> f64 {
        if self.items.len() < 2 {
            return 1.0;
        }
        let first = self.items.first().unwrap().timestamp_ms as f64 / 1000.0;
        let last = self.items.last().unwrap().timestamp_ms as f64 / 1000.0;
        let span = last - first;
        if span <= 0.0 {
            1.0
        } else {
            self.items.len() as f64 / span
        }
    }

    fn timestamp(&self, index: usize) -> u64 {
        self.items[index].timestamp_ms
    }

    fn inflate(&self, index: usize) -> (PromptPayload, u64, u64) {
        let item = &self.items[index];
        (
            PromptPayload::Body(item.body.clone()),
            item.input_length,
            item.output_length,
        )
    }
}
