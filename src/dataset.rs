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
    pub fn load_mooncake_jsonl(path: &str) -> Self {
        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        pub struct MooncakeRecord {
            pub timestamp: u64,
            pub input_length: u64,
            pub output_length: u64,
            pub hash_ids: Vec<u64>,
        }

        let mut requests = Vec::new();

        let file = File::open(path).unwrap();
        for line in BufReader::new(file).lines() {
            let record: MooncakeRecord = serde_json::from_str(line.unwrap().as_str()).unwrap();
            requests.push((record.input_length, record.output_length));
        }

        requests.shuffle(&mut rand::thread_rng());

        Self {
            dataset_size: requests.len(),
            requests,
            next_index: AtomicUsize::new(0),
        }
    }

    pub fn load_burstgpt_csv(path: &str) -> Self {
        let mut requests = Vec::new();
        let mut reader = BufReader::new(File::open(path).unwrap());
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        for line in reader.lines() {
            let parts = line
                .unwrap()
                .split(',')
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let log_type = &parts[5];
            let input_length = parts[2].parse().unwrap();
            let output_length = parts[3].parse().unwrap();
            if log_type == "Conversation log" {
                requests.push((input_length, output_length));
            }
        }

        requests.shuffle(&mut rand::thread_rng());

        Self {
            dataset_size: requests.len(),
            requests,
            next_index: AtomicUsize::new(0),
        }
    }

    pub fn load_mock_dataset() -> Self {
        let requests = (0..1000)
            .into_iter()
            .map(|_| (rand::random::<u64>() % 100, rand::random::<u64>() % 100))
            .collect::<Vec<_>>();

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
        let dataset_path = std::path::Path::new("./data/mooncake_trace.jsonl");
        if dataset_path.exists() {
            let dataset = Dataset::load_mooncake_jsonl(dataset_path.to_str().unwrap());
            for _ in 0..10 {
                println!("(input, output): {:?}", dataset.next_request());
            }
        } else {
            eprintln!("Dataset not found");
        }
    }
}
