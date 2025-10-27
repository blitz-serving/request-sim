use std::{
    cell::UnsafeCell,
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    metrics::{self, SystemMetrics},
    token_sampler::TokenSampler,
    SpinRwLock,
};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use tracing::{instrument, Level}; 

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

fn from_timestamp<'de, D>(deserializer: D) -> Result<NaiveDateTime, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    // 支持 "2023-11-16 18:15:46.6805900" 这种格式
    NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S%.f").map_err(serde::de::Error::custom)
}

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
    fn timestamp(&self, index: usize) -> u64;
    fn inflate(&self, index: usize, ts: &TokenSampler) -> (String, u64, u64, SystemMetrics);
    fn iter(&self) -> DataIter;
    fn rps(&self) -> f64;
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

    fn rps(&self) -> f64 {
        self.items.len() as f64
            / (self.items.last().unwrap().timestamp - self.items.first().unwrap().timestamp)
    }

    fn timestamp(&self, index: usize) -> u64 {
        (self.items[index].timestamp * 1000.) as u64
    }

    #[instrument(skip_all, fields(chat_id = index), level = Level::INFO)]
    fn inflate(&self, index: usize, ts: &TokenSampler) -> (String, u64, u64, SystemMetrics) {
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

            (
                prompt,
                (*data_item).input_length,
                (*data_item).output_length,
                SystemMetrics {
                    generate_time: None,
                    get_prompt_time: None,
                    sample_time: None,
                    inflate_time: None,
                    send_gap: None,
                    prev_sample_time: None,
                    post_sample_time: None,
                },
            )
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

    fn rps(&self) -> f64 {
        self.items.len() as f64
            / (self.items.last().unwrap().timestamp as f64
                - self.items.first().unwrap().timestamp as f64)
    }

    fn timestamp(&self, index: usize) -> u64 {
        self.items[index].timestamp
    }

    fn inflate(&self, index: usize, ts: &TokenSampler) -> (String, u64, u64, SystemMetrics) {
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

            (
                prompt,
                (*data_item).input_length,
                (*data_item).output_length,
                SystemMetrics {
                    generate_time: None,
                    get_prompt_time: None,
                    sample_time: None,
                    inflate_time: None,
                    send_gap: None,
                    prev_sample_time: None,
                    post_sample_time: None,
                },
            )
        }
    }
}

//
// ============== AzureDataset ==============
//
pub struct AzureDataset {
    start_time: u64,
    items: Vec<AzureDataItem>,
    user_prompts: UnsafeCell<Vec<String>>, // each string represents 16 tokens
    rwlock: SpinRwLock,
    // user_prompts_map: UnsafeCell<HashMap<usize, String>>,
}

unsafe impl Send for AzureDataset {}
unsafe impl Sync for AzureDataset {}

impl AzureDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(Vec::with_capacity(1024)),
            rwlock: SpinRwLock::new(),
            start_time: 0,
            // user_prompts_map: UnsafeCell::new(HashMap::new()),
        }
    }
}

impl LLMTrace for AzureDataset {
    fn load(&mut self, path: &str) {
        let mut rdr = csv::Reader::from_path(path).unwrap();
        for result in rdr.deserialize() {
            let mut record: AzureDataItem = result.unwrap();
            if self.start_time == 0 {
                self.start_time = record.naive_timestamp.and_utc().timestamp_millis() as u64;
            }
            record.timestamp =
                record.naive_timestamp.and_utc().timestamp_millis() as u64 - self.start_time;
            self.items.push(record);
        }
    }

    fn iter(&self) -> DataIter {
        DataIter {
            size: self.items.len(),
            index: AtomicUsize::new(0),
        }
    }

    fn rps(&self) -> f64 {
        let first = self.items.first().unwrap().timestamp;
        let last = self.items.last().unwrap().timestamp;
        let duration = last - first;
        let seconds = duration as f64 / 1000.0;
        if seconds > 0.0 {
            self.items.len() as f64 / seconds
        } else {
            0.0
        }
        // self.items.len() as f64
        //     / (self.items.last().unwrap().timestamp - self.items.first().unwrap().timestamp) as f64
    }

    fn timestamp(&self, index: usize) -> u64 {
        // (self.items[index].timestamp * 1000.) as u64
        self.items[index].timestamp
    }

    fn inflate(&self, index: usize, ts: &TokenSampler) -> (String, u64, u64, SystemMetrics) {
        unsafe {
            let inflate_start_time = std::time::Instant::now();
            let mut metrics = SystemMetrics {
                generate_time: None,
                get_prompt_time: None,
                sample_time: None,
                inflate_time: None,
                send_gap: None,
                prev_sample_time: None,
                post_sample_time: None,
            };
            // tracing::info!("Inflating index {}", index);
            let AzureDataItem {
                timestamp: _,
                context_tokens,
                generated_tokens,
                naive_timestamp: _,
            } = self.items.get(index).unwrap().clone();

            let last_block_len = (context_tokens % 16) as usize;
            let num_blocks = (context_tokens as usize - last_block_len) / 16;

            let mut prompt = String::new();
            self.rwlock.read_lock();
            let n = (&*self.user_prompts.get()).len();

            let read_lock_time = inflate_start_time.elapsed().as_millis();
            metrics.prev_sample_time = Some(read_lock_time as u64);
            if n >= num_blocks {
                let get_prompt_start_time = std::time::Instant::now();
                for s in &(&(*self.user_prompts.get()))[0..num_blocks] {
                    prompt.push_str(s);
                }
                // tracing::info!("no need to generate new blocks, current blocks = {}", n);
                self.rwlock.read_unlock();
                let end_time = get_prompt_start_time.elapsed().as_millis();
                metrics.get_prompt_time = Some(end_time as u64);
            } else {
                let generate_start_time = std::time::Instant::now();
                for s in &(&(*self.user_prompts.get()))[0..n] {
                    prompt.push_str(s);
                }
                // tracing::info!(
                //     "need to generate {} new blocks, current blocks = {}",
                //     num_blocks - n,
                //     n
                // );
                self.rwlock.read_unlock();
                let new_prompts: Vec<String> = (n..num_blocks).map(|_| ts.gen_string(16)).collect();
                for s in new_prompts.iter() {
                    prompt.push_str(s);
                }

                // tracing::info!("waiting for write lock, index = {}", index);
                self.rwlock.write_lock();
                (&mut *self.user_prompts.get()).extend(new_prompts);
                self.rwlock.write_unlock();
                let end_time = generate_start_time.elapsed().as_millis();
                metrics.generate_time = Some(end_time as u64);
            }
            // postcond: self.rwlock is unlocked
            // tracing::info!("generating last block of length {}", last_block_len);
            let post_sample_time = std::time::Instant::now();
            if last_block_len != 0 {
                let last_block_prompt = ts.gen_string(last_block_len);
                prompt.push_str(&last_block_prompt);
            }
            // self.rwlock.read_lock();
            // if let Some(s) = &(&(*self.user_prompts_map.get())).get(&last_block_len) {
            //     prompt.push_str(s);
            //     self.rwlock.read_unlock();
            // } else {
            //     self.rwlock.read_unlock();
            //     let last_block_prompt = ts.gen_string(last_block_len);
            //     prompt.push_str(&last_block_prompt);
            //     self.rwlock.write_lock();
            //     (&mut *self.user_prompts_map.get()).insert(last_block_len, last_block_prompt);
            //     self.rwlock.write_unlock();
            // }

            let end_time = inflate_start_time.elapsed().as_millis();
            metrics.inflate_time = Some(end_time as u64);
            metrics.post_sample_time = Some(post_sample_time.elapsed().as_millis() as u64);
            (prompt, context_tokens, generated_tokens, metrics)
        }
    }
}
#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::token_sampler::TokenSampler;
    // use tokenizers::Tokenizer;

    // #[test]
    // fn test_bailian_dataset() {
    //     let path = "./data/test/bailian.jsonl";
    //     let mut ds = BailianDataset::new();
    //     ds.load(path);

    //     let tokenizer = Tokenizer::from_file("./data/test/tokenizer.json").unwrap();

    //     let ts = TokenSampler::new(tokenizer);
    //     let iter = ds.iter();

    //     for idx in iter {
    //         let (prompt, in_len, out_len) = ds.inflate(idx, &ts);
    //         println!(
    //             "Bailian prompt = \"{}\", output length = {}",
    //             prompt, out_len
    //         );
    //     }
    // }

    // #[test]
    // fn test_mooncake_dataset() {
    //     let path = "./data/test/mooncake.jsonl";
    //     let mut ds = MooncakeDataset::new();
    //     ds.load(path);

    //     let tokenizer = Tokenizer::from_file("./data/test/tokenizer.json").unwrap();

    //     let ts = TokenSampler::new(tokenizer);
    //     let iter = ds.iter();

    //     for idx in iter {
    //         let (prompt, in_len, out_len) = ds.inflate(idx, &ts);
    //         println!(
    //             "Mooncake prompt = \"{}\", output length = {}",
    //             prompt, out_len
    //         );
    //     }
    // }

    // #[test]
    // fn test_azure_dataset() {
    //     let path = "./data/test/azure.csv";
    //     let mut ds = AzureDataset::new();
    //     ds.load(path);

    //     let tokenizer = Tokenizer::from_file("./data/test/tokenizer.json").unwrap();

    //     let ts = TokenSampler::new(tokenizer);
    //     let iter = ds.iter();

    //     for idx in iter {
    //         let (prompt, in_len, out_len) = ds.inflate(idx, &ts);
    //         println!("Azure prompt = \"{}\", output length = {}", prompt, out_len);
    //     }
    // }
}
