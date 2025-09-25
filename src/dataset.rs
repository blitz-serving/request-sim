use std::{
    cell::UnsafeCell,
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    mem,
    sync::atomic::{AtomicUsize, Ordering},
};

use chrono::NaiveDateTime;
use rand::seq::SliceRandom;
use regex::Regex;
use crate::{token_sampler::TokenSampler, SpinRwLock};
use serde::{Deserialize, Serialize};

/// jsonl of Bailian
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MooncakeDataItem {
    pub timestamp: u64,
    pub input_length: u64,
    pub output_length: u64,
    pub hash_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AzureDataItem {
    #[serde(rename = "TIMESTAMP")]
    pub timestamp: f32,
    #[serde(rename = "ContextTokens")]
    pub context_tokens: u64,
    #[serde(rename = "GeneratedTokens")]
    pub generated_tokens: u64,
}

pub struct DataIter {
    size: usize,
    index: AtomicUsize,
}

impl Iterator for DataIter {
    type Item = *const u8;
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
    fn timestamp(&self, index: usize) -> u64;
    fn inflate(&self, index: usize, ts: &TokenSampler) -> (String, u64);
    fn iter(&self) -> DataIter;
}

//
// ============== BailianDataset ==============
//
pub struct BailianDataset {
    items: Vec<BailianDataItem>,
    user_prompts: UnsafeCell<HashMap<u64, String>>,
    rwlock: SpinRwLock,
}

impl BailianDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(HashMap::new()),
            rwlock: SpinRwLock::new(),
        }
    }
}

unsafe impl Send for BailianDataset {}
unsafe impl Sync for BailianDataset {}

impl LLMTrace for BailianDataset {
    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();

        for line in BufReader::new(file).lines() {
            let item: BailianDataItem = serde_json::from_str(line.unwrap().as_str()).unwrap();
            self.items.push(item);
        }
    }

    fn iter(&self) -> DataIter {
        DataIter {
            size: self.items.len(),
            index: AtomicUsize::new(0),
        }
    }

    fn timestamp(&self, index: usize) -> u64 {
        (self.items[index].timestamp * 1000.) as u64
    }

    fn inflate(&self, index: usize, ts: &TokenSampler) -> (String, u64) {
        // NOTE: the last block hash may be hashed onto a partially filled block
        const BLOCK_SIZE: usize = 16;
        unsafe {
            let data_item = self.items.get(index).unwrap();
            let last_block_len =
                (*data_item).input_length as usize - ((*data_item).hash_ids.len() - 1) * BLOCK_SIZE;
            debug_assert!(last_block_len <= BLOCK_SIZE);

            let mut prompt = String::new();
            for &hash_id in (*data_item)
                .hash_ids
                .iter()
                .take((*data_item).hash_ids.len() - 1)
            {
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

            let last_block_prompt = ts.gen_string(last_block_len);
            prompt.push_str(&last_block_prompt);
            self.rwlock.write_lock();
            (&mut *self.user_prompts.get())
                .insert(*(*data_item).hash_ids.last().unwrap(), last_block_prompt);
            self.rwlock.write_unlock();

            (prompt, (*data_item).output_length)
        }
    }
}

//
// ============== MooncakeDataset ==============
//
pub struct MooncakeDataset {
    items: Vec<MooncakeDataItem>,
    user_prompts: UnsafeCell<HashMap<u64, String>>,
    rwlock: SpinRwLock,
}

unsafe impl Send for MooncakeDataset {}
unsafe impl Sync for MooncakeDataset {}

impl MooncakeDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(HashMap::new()),
            rwlock: SpinRwLock::new(),
        }
    }
}

impl LLMTrace for MooncakeDataset {
    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        for line in BufReader::new(file).lines() {
            let item: MooncakeDataItem = serde_json::from_str(line.unwrap().as_str()).unwrap();
            self.items.push(item);
        }
    }

    fn iter(&self) -> DataIter {
        DataIter {
            size: self.items.len(),
            index: AtomicUsize::new(0),
        }
    }

    fn timestamp(&self, index: usize) -> u64 {
        self.items[index].timestamp   
    }

    fn inflate(&self, index: usize, ts: &TokenSampler) -> (String, u64) {
        // NOTE: the last block hash may be hashed onto a partially filled block
        const BLOCK_SIZE: usize = 512;
        unsafe {
            let data_item = self.items.get(index).unwrap();
            let last_block_len =
                (*data_item).input_length as usize - ((*data_item).hash_ids.len() - 1) * BLOCK_SIZE;
            debug_assert!(last_block_len <= BLOCK_SIZE);

            let mut prompt = String::new();
            for &hash_id in (*data_item)
                .hash_ids
                .iter()
                .take((*data_item).hash_ids.len() - 1)
            {
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

            let last_block_prompt = ts.gen_string(last_block_len);
            prompt.push_str(&last_block_prompt);
            self.rwlock.write_lock();
            (&mut *self.user_prompts.get())
                .insert(*(*data_item).hash_ids.last().unwrap(), last_block_prompt);
            self.rwlock.write_unlock();

            (prompt, (*data_item).output_length)
        }
    }
}

//
// ============== AzureDataset ==============
//
pub struct AzureDataset {
    items: Vec<AzureDataItem>,
    user_prompts: UnsafeCell<Vec<String>>, // each string represents 16 tokens
    rwlock: SpinRwLock,
}

unsafe impl Send for AzureDataset {}
unsafe impl Sync for AzureDataset {}

impl AzureDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(Vec::with_capacity(1024)),
            rwlock: SpinRwLock::new(),
        }
    }
}

impl LLMTrace for AzureDataset {
    fn load(&mut self, path: &str) {
        let mut rdr = csv::Reader::from_path(path).unwrap();
        for result in rdr.deserialize() {
            let record: AzureDataItem = result.unwrap();
            self.items.push(record);
        }
    }

    fn iter(&self) -> DataIter {
        DataIter {
            size: self.items.len(),
            index: AtomicUsize::new(0),
        }
    }

    fn timestamp(&self, index: usize) -> u64 {
        (self.items[index].timestamp * 1000.) as u64
    }

    fn inflate(&self, index: usize, ts: &TokenSampler) -> (String, u64) {
        unsafe {
            let AzureDataItem {
                timestamp: _,
                context_tokens,
                generated_tokens,
            } = self.items.get(index).unwrap().clone();

            let last_block_len = (context_tokens % 16) as usize;
            let num_blocks = (context_tokens as usize - last_block_len) / 16;

            let mut prompt = String::new();
            self.rwlock.read_lock();
            let n = (&*self.user_prompts.get()).len();
            if n >= num_blocks {
                for s in &(&(*self.user_prompts.get()))[0..num_blocks] {
                    prompt.push_str(s);
                }
                self.rwlock.read_unlock();
            } else {
                for s in &(&(*self.user_prompts.get()))[0..n] {
                    prompt.push_str(s);
                }
                self.rwlock.read_unlock();
                let new_prompts: Vec<String> = (n..num_blocks).map(|_| ts.gen_string(16)).collect();
                for s in new_prompts.iter() {
                    prompt.push_str(s);
                }
                self.rwlock.write_lock();
                (&mut *self.user_prompts.get()).extend(new_prompts);
                self.rwlock.write_unlock();
            }
            // postcond: self.rwlock is unlocked

            let last_block_prompt = ts.gen_string(last_block_len);
            prompt.push_str(&last_block_prompt);

            (prompt, generated_tokens)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_sampler::TokenSampler;
    use tokenizers::Tokenizer;

    #[test]
    fn test_bailian_dataset() {
        let path = "./data/test/bailian.jsonl";
        let mut ds = BailianDataset::new();
        ds.load(path);

        let tokenizer = Tokenizer::from_file("./data/test/tokenizer.json").unwrap();

        let ts = TokenSampler::new(tokenizer);
        let iter = ds.iter();

        for p in iter {
            let (prompt, out_len) = ds.inflate(p, &ts);
            println!(
                "Bailian prompt = \"{}\", output length = {}",
                prompt, out_len
            );
        }
    }

    #[test]
    fn test_mooncake_dataset() {
        let path = "./data/test/mooncake.jsonl";
        let mut ds = MooncakeDataset::new();
        ds.load(path);

        let tokenizer = Tokenizer::from_file("./data/test/tokenizer.json").unwrap();

        let ts = TokenSampler::new(tokenizer);
        let iter = ds.iter();

        for p in iter {
            let (prompt, out_len) = ds.inflate(p, &ts);
            println!(
                "Mooncake prompt = \"{}\", output length = {}",
                prompt, out_len
            );
        }
    }

    #[test]
    fn test_azure_dataset() {
        let path = "./data/test/azure.csv";
        let mut ds = AzureDataset::new();
        ds.load(path);

        let tokenizer = Tokenizer::from_file("./data/test/tokenizer.json").unwrap();

        let ts = TokenSampler::new(tokenizer);
        let iter = ds.iter();

        for p in iter {
            let (prompt, out_len) = ds.inflate(p, &ts);
            println!("Azure prompt = \"{}\", output length = {}", prompt, out_len);
        }
    }
}
