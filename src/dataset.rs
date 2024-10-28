use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::AtomicUsize,
};

use chrono::NaiveDateTime;
use rand::seq::SliceRandom;
use regex::Regex;

pub struct Dataset {
    dataset_size: usize,

    /// Each request is a tuple of (input_length, output_length).
    requests: Vec<(u64, u64)>,

    timestamps: Vec<u64>,

    /// The time it takes to complete a round of requests in milliseconds.
    round_time: u64,

    next: AtomicUsize,
}

impl Dataset {
    /// If `shuffle` is true, the dataset will be shuffled and the requests returned by [`Self::next_request`] is in random order.
    pub fn load_mooncake_jsonl(
        path: &str,
        shuffle: bool,
        prefill_only: bool,
        filter_long_context: bool,
    ) -> Self {
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

        if prefill_only {
            for line in BufReader::new(file).lines() {
                let record: MooncakeRecord = serde_json::from_str(line.unwrap().as_str()).unwrap();
                if !filter_long_context && record.input_length + record.output_length >= 4096 {
                    requests.push((4095 - record.output_length, record.output_length));
                } else {
                    requests.push((record.input_length, 1));
                }
                timestamps.push(record.timestamp);
            }
        } else {
            for line in BufReader::new(file).lines() {
                let record: MooncakeRecord = serde_json::from_str(line.unwrap().as_str()).unwrap();
                if !filter_long_context && record.input_length + record.output_length >= 4096 {
                    requests.push((4095 - record.output_length, record.output_length));
                } else {
                    requests.push((record.input_length, record.output_length));
                }
                timestamps.push(record.timestamp);
            }
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
    pub fn load_burstgpt_csv(
        path: &str,
        shuffle: bool,
        prefill_only: bool,
        filter_long_context: bool,
    ) -> Self {
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
            let mut input_length = parts[2].parse().unwrap();
            let output_length = if prefill_only {
                1
            } else {
                parts[3].parse().unwrap()
            };
            if !filter_long_context && input_length + output_length >= 4096 {
                input_length = 4095 - output_length;
            }
            let log_type = &parts[5];
            if log_type == "Conversation log" || log_type == "API log" {
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

    // ts: mooncake, input&output length: burstgpt
    pub fn load_mooncake_ts_burst_data(
        mooncake_path: &str,
        burstgpt_path: &str,
        shuffle: bool,
        prefill_only: bool,
    ) -> Self {
        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        pub struct MoonCakeInfoRecord {
            pub timestamp: u64,
            pub input_length: u64,
            pub output_length: u64,
            pub hash_ids: Vec<u64>,
        }

        let mut requests = Vec::new();
        let mut timestamps = Vec::new();

        let file_mooncake = File::open(mooncake_path).unwrap();
        for line in BufReader::new(file_mooncake).lines() {
            let record: MoonCakeInfoRecord = serde_json::from_str(line.unwrap().as_str()).unwrap();
            timestamps.push(record.timestamp);
        }
        // fill requests with burstgpt data, each mooncake request has a corresponding burstgpt request
        let file_burstgpt = File::open(burstgpt_path).unwrap();
        let mut reader = BufReader::new(file_burstgpt);
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        let mut burstgpt_requests = Vec::new();
        for line in reader.lines() {
            let parts = line
                .unwrap()
                .split(',')
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let input_length = parts[2].parse().unwrap();
            let output_length = if prefill_only {
                1
            } else {
                parts[3].parse().unwrap()
            };
            let log_type = &parts[5];
            if log_type == "Conversation log" {
                burstgpt_requests.push((input_length, output_length));
            }
        }
        let mut burstgpt_iter = burstgpt_requests.iter();
        for _ in timestamps.iter() {
            let (input_length, output_length) = burstgpt_iter.next().unwrap();
            requests.push((*input_length, *output_length));
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

    pub fn load_azure_csv(
        path: &str,
        shuffle: bool,
        prefill_only: bool,
        filter_long_context: bool,
    ) -> Self {
        let mut requests = Vec::new();
        let mut timestamps = Vec::new();
        let mut reader = BufReader::new(File::open(path).unwrap());

        // skip header
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();

        let mut initial_timestamp: Option<NaiveDateTime> = None;

        for line in reader.lines() {
            let parts = line
                .unwrap()
                .split(',')
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let timestamp_str = &parts[0];
            let timestamp =
                NaiveDateTime::parse_from_str(&timestamp_str, "%Y-%m-%d %H:%M:%S%.f").unwrap();
            if initial_timestamp.is_none() {
                initial_timestamp = Some(timestamp);
            }
            let elapsed = timestamp - initial_timestamp.unwrap();
            let elapsed_millis = elapsed.num_milliseconds() as u64;

            let mut input_length = parts[1].parse().unwrap();
            let output_length = if prefill_only {
                1
            } else {
                parts[2].parse().unwrap()
            };
            if !filter_long_context && input_length + output_length >= 4096 {
                input_length = 4095 - output_length;
            }
            timestamps.push(elapsed_millis);
            requests.push((input_length, output_length));
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

    pub fn cherry_pick_burstgpt(
        path: &str,
        shuffle: bool,
        prefill_only: bool,
        filter_long_context: bool,
        start_ts: u64,
        end_ts: u64,
    ) -> Self {
        let mut requests = Vec::new();
        let mut timestamps = Vec::new();
        let mut reader = BufReader::new(File::open
        (path).unwrap());

        // skip header
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();

        let mut initial_timestamp: Option<u64> = None;

        for line in reader.lines() {
            let parts = line
                .unwrap()
                .split(',')
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let timestamp = parts[0].parse::<f64>().unwrap() as u64;
            if initial_timestamp.is_none() {
                initial_timestamp = Some(timestamp);
            }
            if timestamp - initial_timestamp.unwrap() < start_ts {
                continue;
            }
            if timestamp - initial_timestamp.unwrap() > end_ts {
                break;
            }
            let mut input_length = parts[2].parse().unwrap();
            let output_length = if prefill_only {
                1
            } else {
                parts[3].parse().unwrap()
            };
            if !filter_long_context && input_length + output_length >= 4096 {
                input_length = 4095 - output_length;
            }
            let log_type = &parts[5];
            if log_type == "Conversation log" || log_type == "API log" {
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
            .map(|_| {
                (
                    rand::random::<u64>() % 100 + 1,
                    rand::random::<u64>() % 100 + 1,
                )
            })
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

    pub fn load_uniform_dataset(input_length: u64, output_length: u64) -> Self {
        let dataset_size = 1000;
        let requests = vec![(input_length, output_length); dataset_size];
        let timestamps = (0..dataset_size)
            .into_iter()
            .map(|i| i as u64 * 1000)
            .collect::<Vec<_>>();
        let round_time = 1000000;

        Self {
            dataset_size,
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

    pub fn request_rate(&self) -> f64 {
        self.dataset_size() as f64 * 1000.0 / self.round_time() as f64
    }
}

pub fn parse_dataset_type(s: &str) -> Result<DatasetType, String> {
    fn parse_u64(text: &str) -> Result<Vec<u64>, String> {
        let mut integers = Vec::new();
        let re = Regex::new(r"\b\d+\b").map_err(|e| e.to_string())?;
        for cap in re.captures_iter(text) {
            integers.push(cap[0].parse::<u64>().map_err(|e| e.to_string())?);
        }
        Ok(integers)
    }

    let s = s.to_lowercase();
    if s.starts_with("uniform") {
        let integers = parse_u64(s.as_str())?;
        Ok(DatasetType::Uniform {
            input: integers[0],
            output: integers[1],
        })
    }
    else if s.starts_with("cherry_pick_burstgpt") {
        let integers = parse_u64(s.as_str())?;
        Ok(DatasetType::CherryPickBurstgpt {
            start_ts: integers[0],
            end_ts: integers[1],
        })
    }else {
        match s.as_ref() {
            "mooncake" => Ok(DatasetType::Mooncake),
            "burstgpt" => Ok(DatasetType::Burstgpt),
            "azure" => Ok(DatasetType::Azure),
            "mooncake_sampled" => Ok(DatasetType::MooncakeSampled),
            "mock" => Ok(DatasetType::Mock),
            _ => Err("Invalid dataset type.".to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DatasetType {
    Mooncake,
    Burstgpt,
    Azure,
    MooncakeSampled,
    Mock,
    Uniform { input: u64, output: u64 },
    CherryPickBurstgpt { start_ts: u64, end_ts: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_mooncake() {
        let dataset_path = std::path::Path::new("./data/mooncake_trace.jsonl");
        if dataset_path.exists() {
            let dataset =
                Dataset::load_mooncake_jsonl(dataset_path.to_str().unwrap(), false, false, false);
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
            let dataset = Dataset::load_burstgpt_csv(dataset_path.to_str().unwrap(), false, false, false);
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
            println!("{}", dataset.next_request_with_timestamp().0);
        }
    }

    #[test]
    fn test_load_azure() {
        let dataset_path = std::path::Path::new("/nvme/ly/dataset/AzureLLMInferenceTrace_code.csv");
        if dataset_path.exists() {
            let dataset = Dataset::load_azure_csv(dataset_path.to_str().unwrap(), false, false, false);
            for _ in 0..10 {
                println!(
                    "(timestamp, input, output): {:?}",
                    dataset.next_request_with_timestamp()
                );
            }
        } else {
            eprintln!("Dataset not found");
        }
    }

    #[test]
    fn test_load_uniform_dataset() {
        let dataset_type = parse_dataset_type("Uniform(10,1)").unwrap();
        println!("{:?}", dataset_type);
        let dataset = Dataset::load_uniform_dataset(10, 1);
        println!(
            "{} {} {}",
            dataset.request_rate(),
            dataset.round_time(),
            dataset.dataset_size()
        );
    }

    #[test]
    fn test_cherry_pick_burstgpt() {
        let dataset_type = parse_dataset_type("cherry_pick_burstgpt(0, 1000)").unwrap();
        println!("{:?}", dataset_type);
        let dataset_path = std::path::Path::new("/nvme/wht/dataset/BurstGPT_without_fails_2.csv");
        if dataset_path.exists() {
            let dataset = Dataset::cherry_pick_burstgpt(
                dataset_path.to_str().unwrap(),
                false,
                false,
                false,
                0,
                1000,
            );
            for _ in 0..10 {
                println!("(input, output): {:?}", dataset.next_request());
            }
        } else {
            eprintln!("Dataset not found");
        }
    }
}
