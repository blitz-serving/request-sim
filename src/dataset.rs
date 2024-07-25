use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::AtomicUsize,
};

use rand::seq::SliceRandom;

pub struct Dataset {
    dataset_size: usize,
    requests: Vec<(u64, u64)>,
    next_index: AtomicUsize,
}

impl Dataset {
    pub fn load_mooncake_trace_jsonl(path: &str) -> Self {
        #[derive(serde::Deserialize)]
        pub struct MooncakeLine {
            pub timestamp: u64,
            pub input_length: u64,
            pub output_length: u64,
            pub hash_ids: Vec<u64>,
        }

        let mut requests = Vec::new();

        let file = File::open(path).unwrap();
        for line in BufReader::new(file).lines() {
            let line: MooncakeLine = serde_json::from_str(line.unwrap().as_str()).unwrap();
            requests.push((line.input_length, line.output_length));
        }

        requests.shuffle(&mut rand::thread_rng());

        Self {
            dataset_size: requests.len(),
            requests,
            next_index: AtomicUsize::new(0),
        }
    }

    pub fn next_request(&self) -> (u64, u64) {
        let index = self
            .next_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.requests[index % self.dataset_size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_mooncake() {
        let dataset = Dataset::load_mooncake_trace_jsonl("mooncake_trace.jsonl");
        for _ in 0..10 {
            let (input_length, output_length) = dataset.next_request();
            println!(
                "input_length: {}, output_length: {}",
                input_length, output_length
            );
        }
    }
}
