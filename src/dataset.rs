use std::{
    cell::UnsafeCell,
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{token_sampler::TokenSampler, SpinLock};
use rand::{seq::SliceRandom, thread_rng, Rng};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

pub struct Dataset {
    dataset_size: usize,
    requests: Vec<(u64, u64)>,
    next_index: AtomicUsize,
}

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
    pub input_length: u64,
    pub output_length: u64,
}

pub(crate) struct DataIter<T> {
    data: *const Vec<T>, // data never moves!
    index: AtomicUsize,
}

impl<T> Iterator for DataIter<T> {
    type Item = *const u8;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let mut i = self.index.fetch_add(1, Ordering::AcqRel);
            if i == (*self.data).len() {
                self.index.store(0, Ordering::Release);
                i = 0;
            }
            let p: *const T = &(*self.data)[i];
            Some(p as *const u8)
        }
    }
}

pub(crate) trait LLMTrace {
    fn load(&mut self, path: &str);
    fn inflate(&self, item: *const u8, ts: &TokenSampler) -> String;
    type DataItem;
    fn iter(&self) -> DataIter<Self::DataItem>;
}

//
// ============== BailianDataset ==============
//
pub(crate) struct BailianDataset {
    items: Vec<BailianDataItem>,
    user_prompts: UnsafeCell<HashMap<u64, String>>,
    mtx: SpinLock,
}

impl BailianDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(HashMap::new()),
            mtx: SpinLock::new(),
        }
    }
}

unsafe impl Send for BailianDataset {}
unsafe impl Sync for BailianDataset {}

impl LLMTrace for BailianDataset {
    type DataItem = BailianDataItem;

    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        for line in BufReader::new(file).lines() {
            let item: BailianDataItem = serde_json::from_str(line.unwrap().as_str()).unwrap();
            self.items.push(item);
        }
    }

    fn iter(&self) -> DataIter<BailianDataItem> {
        DataIter::<BailianDataItem> {
            data: &self.items,
            index: AtomicUsize::new(0),
        }
    }

    fn inflate(&self, item: *const u8, ts: &TokenSampler) -> String {
        unsafe {
            let data_item = item as *mut BailianDataItem;
            let last_block_len =
                (*data_item).input_length as usize - (*data_item).hash_ids.len() * 16;
            debug_assert!(last_block_len <= 16);

            let mut prompt = String::new();
            for &hash_id in (*data_item).hash_ids.iter() {
                self.mtx.lock();
                if let Some(s) = (&*self.user_prompts.get()).get(&hash_id) {
                    prompt.push_str(&s);
                } else {
                    let s = ts.gen_string(16);
                    prompt.push_str(&s);
                    (&mut *self.user_prompts.get()).insert(hash_id, s);
                }
                self.mtx.unlock();
            }

            prompt
        }
    }
}

//
// ============== MooncakeDataset ==============
//
pub(crate) struct MooncakeDataset {
    items: Vec<MooncakeDataItem>,
    user_prompts: UnsafeCell<HashMap<u64, String>>,
    mtx: SpinLock,
}

unsafe impl Send for MooncakeDataset {}
unsafe impl Sync for MooncakeDataset {}

impl MooncakeDataset {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            user_prompts: UnsafeCell::new(HashMap::new()),
            mtx: SpinLock::new(),
        }
    }
}

impl LLMTrace for MooncakeDataset {
    type DataItem = MooncakeDataItem;

    fn load(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        for line in BufReader::new(file).lines() {
            let item: MooncakeDataItem =
                serde_json::from_str(line.unwrap().as_str()).unwrap();
            self.items.push(item);
        }
    }

    fn iter(&self) -> DataIter<MooncakeDataItem> {
        DataIter::<MooncakeDataItem> {
            data: &self.items,
            index: AtomicUsize::new(0),
        }
    }

    fn inflate(&self, item: *const u8, ts: &TokenSampler) -> String {
        unsafe {
            let data_item = item as *mut MooncakeDataItem;
            let mut prompt = String::new();
            for &hash_id in (*data_item).hash_ids.iter() {
                self.mtx.lock();
                if let Some(s) = (&*self.user_prompts.get()).get(&hash_id) {
                    prompt.push_str(&s);
                } else {
                    let s = ts.gen_string(16);
                    prompt.push_str(&s);
                    (&mut *self.user_prompts.get()).insert(hash_id, s);
                }
                self.mtx.unlock();
            }
            prompt
        }
    }
}

//
// ============== AzureDataset ==============
//
pub(crate) struct AzureDataset {
    items: Vec<AzureDataItem>,
}

unsafe impl Send for AzureDataset {}
unsafe impl Sync for AzureDataset {}

impl AzureDataset {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
}

impl LLMTrace for AzureDataset {
    type DataItem = AzureDataItem;

    fn load(&mut self, path: &str) {
        let mut rdr = csv::Reader::from_path(path).unwrap();
        for result in rdr.deserialize() {
            let record: AzureDataItem = result.unwrap();
            self.items.push(record);
        }
    }

    fn iter(&self) -> DataIter<AzureDataItem> {
        DataIter::<AzureDataItem> {
            data: &self.items,
            index: AtomicUsize::new(0),
        }
    }

    fn inflate(&self, item: *const u8, _ts: &TokenSampler) -> String {
        unsafe {
            let data_item = item as *mut AzureDataItem;
            format!(
                "[Azure] in={} out={}",
                (*data_item).input_length,
                (*data_item).output_length
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    struct DummySampler;
    impl DummySampler {
        fn new() -> Self { DummySampler }
        fn gen_string(&self, n: usize) -> String {
            format!("dummy_{n}")
        }
    }

    #[test]
    fn test_bailian_dataset() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(
            tmp,
            r#"{{"chat_id":1,"parent_chat_id":0,"timestamp":0.0,"input_length":16,"output_length":4,"type":"test","turn":1,"hash_ids":[42]}}"#
        )
        .unwrap();

        let mut ds = BailianDataset::new();
        ds.load(tmp.path().to_str().unwrap());

        let mut it = ds.iter();
        let sampler = DummySampler::new();

        let item_ptr = it.next().unwrap();
        let s = ds.inflate(item_ptr, &sampler);
        assert!(s.contains("dummy_16"));
    }

    #[test]
    fn test_mooncake_dataset() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(
            tmp,
            r#"{{"timestamp":123,"input_length":16,"output_length":8,"hash_ids":[7,8]}}"#
        )
        .unwrap();

        let mut ds = MooncakeDataset::new();
        ds.load(tmp.path().to_str().unwrap());

        let mut it = ds.iter();
        let sampler = DummySampler::new();

        let item_ptr = it.next().unwrap();
        let s = ds.inflate(item_ptr, &sampler);
        assert!(s.contains("dummy_16"));
    }

    #[test]
    fn test_azure_dataset() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "input_length,output_length").unwrap();
        writeln!(tmp, "10,20").unwrap();

        let mut ds = AzureDataset::new();
        ds.load(tmp.path().to_str().unwrap());

        let mut it = ds.iter();
        let sampler = DummySampler::new();

        let item_ptr = it.next().unwrap();
        let s = ds.inflate(item_ptr, &sampler);
        assert!(s.contains("in=10"));
        assert!(s.contains("out=20"));
    }
}