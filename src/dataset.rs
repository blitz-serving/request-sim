use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::AtomicUsize,
};

use rand::seq::SliceRandom;

pub struct Dataset {
    dataset_size: usize,

    /// Each request is a tuple of (input_length, output_length).
    requests: Vec<(u64, u64)>,

    timestamps: Vec<u64>,

    round_time: u64,

    next: AtomicUsize,
}

impl Dataset {
    /// If `shuffle` is true, the dataset will be shuffled and the requests returned by [`Self::next_request`] is in random order.
    pub fn load_mooncake_jsonl(path: &str, shuffle: bool) -> Self {
        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        pub struct MooncakeRecord {
            pub timestamp: u64,
            pub input_length: u64,
            pub output_length: u64,
            pub hash_ids: Vec<u64>,
        }

        let mut requests = Vec::new();
        let mut timestamps = Vec::new();

        let file = File::open(path).unwrap();
        for line in BufReader::new(file).lines() {
            let record: MooncakeRecord = serde_json::from_str(line.unwrap().as_str()).unwrap();
            requests.push((record.input_length, record.output_length));
            timestamps.push(record.timestamp);
        }

        if shuffle {
            requests.shuffle(&mut rand::thread_rng());
        }

        let round_time =
            timestamps.last().unwrap() + timestamps.last().unwrap() / (requests.len() as u64 - 1);

        Self {
            dataset_size: requests.len(),
            round_time,
            requests,
            timestamps,
            next: AtomicUsize::new(0),
        }
    }

    /// If `shuffle` is true, the dataset will be shuffled and the requests returned by [`Self::next_request`] is in random order.
    pub fn load_burstgpt_csv(path: &str, shuffle: bool) -> Self {
        let mut requests = Vec::new();
        let mut timestamps = Vec::new();
        let mut reader = BufReader::new(File::open(path).unwrap());

        // Skip header
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();

        for line in reader.lines() {
            let parts = line
                .unwrap()
                .split(',')
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let timestamp = parts[0].parse::<f64>().unwrap() as u64;
            let input_length = parts[2].parse().unwrap();
            let output_length = parts[3].parse().unwrap();
            let log_type = &parts[5];
            if log_type == "Conversation log" {
                timestamps.push(timestamp);
                requests.push((input_length, output_length));
            }
        }

        let base_timestamp = timestamps[0];
        timestamps
            .iter_mut()
            .for_each(|timestamp| *timestamp = (*timestamp - base_timestamp) * 1000);

        if shuffle {
            requests.shuffle(&mut rand::thread_rng());
        }

        let round_time =
            timestamps.last().unwrap() + timestamps.last().unwrap() / (requests.len() as u64 - 1);

        Self {
            dataset_size: requests.len(),
            round_time,
            requests,
            timestamps,
            next: AtomicUsize::new(0),
        }
    }

    pub fn load_mock_dataset() -> Self {
        let requests = (0..1000)
            .into_iter()
            .map(|_| (rand::random::<u64>() % 100, rand::random::<u64>() % 100))
            .collect::<Vec<_>>();
        let timestamps = (0..1000).into_iter().map(|i| i * 1000).collect::<Vec<_>>();
        let round_time =
            timestamps.last().unwrap() + timestamps.last().unwrap() / (requests.len() as u64 - 1);

        Self {
            dataset_size: requests.len(),
            requests,
            timestamps,
            round_time,
            next: AtomicUsize::new(0),
        }
    }

    pub fn next_request(&self) -> (u64, u64) {
        let next_index = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.requests[next_index % self.dataset_size]
    }

    /// Returned tuple is (timestamp, input_length, output_length).
    pub fn next_request_with_timestamp(&self) -> (u64, u64, u64) {
        let next_index = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let round = next_index / self.dataset_size;
        let index = next_index % self.dataset_size;

        let (input, output) = self.requests[next_index % self.dataset_size];
        let ts = round as u64 * self.round_time + self.timestamps[index % self.dataset_size];
        (ts, input, output)
    }

    pub fn dataset_size(&self) -> usize {
        self.dataset_size
    }

    pub fn round_time(&self) -> u64 {
        self.round_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_mooncake() {
        let dataset_path = std::path::Path::new("./data/mooncake_trace.jsonl");
        if dataset_path.exists() {
            let dataset = Dataset::load_mooncake_jsonl(dataset_path.to_str().unwrap(), false);
            for _ in 0..10 {
                println!("(input, output): {:?}", dataset.next_request());
            }
        } else {
            eprintln!("Dataset not found");
        }
    }

    #[test]
    fn test_load_burstgpt() {
        let dataset_path = std::path::Path::new("./data/BurstGPT_without_fails_2.csv");
        if dataset_path.exists() {
            let dataset = Dataset::load_burstgpt_csv(dataset_path.to_str().unwrap(), false);
            for _ in 0..10 {
                println!("(input, output): {:?}", dataset.next_request());
            }
        } else {
            eprintln!("Dataset not found");
        }
    }

    #[test]
    fn test_dataset_timestamp() {
        let dataset = Dataset::load_mock_dataset();
        let dataset_size = dataset.dataset_size();
        for _ in 0..(dataset_size - 5) {
            dataset.next_request_with_timestamp();
        }
        for _ in 0..10 {
            println!("{}", dataset.next_request_with_timestamp());
        }
    }
}
