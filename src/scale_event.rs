use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::{AtomicUsize, Ordering},
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScaleEventType {
    TriggerPrefillUp,
    TriggerDecodeUp,
    TriggerNormalUp,
    TriggerMutation,
    TriggerScaleDown,
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScaleEvent {
    events: Vec<(u64, usize, usize, ScaleEventType)>,
    next: AtomicUsize,
    pub event_size: usize,
    round_time: u64,
}

impl ScaleEvent {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next: AtomicUsize::new(0),
            event_size: 0,
            round_time: 0,
        }
    }

    pub fn parse_event_csv(&mut self, file_path: &str) {
        let mut reader = BufReader::new(File::open(file_path).unwrap());
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        let mut events = Vec::new();

        for line in reader.lines() {
            let line = line.unwrap();
            let mut iter = line.split(",");
            let event = iter.next().unwrap().to_string();
            let src = iter.next().unwrap().parse::<usize>().unwrap();
            let dst = iter.next().unwrap().parse::<usize>().unwrap();
            let s_time = iter.next().unwrap().parse::<u64>().unwrap();
            let scale_event: ScaleEventType;
            scale_event = if event.contains("prefill_up") {
                ScaleEventType::TriggerPrefillUp
            } else if event.contains("down") {
                ScaleEventType::TriggerScaleDown
            } else if event.contains("decode_up") {
                ScaleEventType::TriggerDecodeUp
            } else if event.contains("mutate") {
                ScaleEventType::TriggerMutation
            } else if event.contains("normal_up") {
                ScaleEventType::TriggerNormalUp
            } else {
                ScaleEventType::Unknown
            };
            events.push((s_time, src, dst, scale_event));
            self.event_size += 1;
            self.round_time = s_time;
        }
        self.events = events;
    }

    pub fn next_event_with_timestamp(&self) -> (u64, usize, usize, ScaleEventType) {
        let next_index = self.next.fetch_add(1, Ordering::Relaxed);
        let round = next_index / self.event_size;
        let index = next_index % self.event_size;
        let (s_time, src, dst, scale_type) = self.events[index];
        let ts = round as u64 * self.round_time + s_time;
        (ts, src, dst, scale_type)
    }

    pub fn request_json_body(&self, src: usize, dst: usize, event_type: ScaleEventType) -> String {
        // assert_ne!(event_type, ScaleEventType::Unknown);
        tracing::info!("src: {}, dst: {}, event_type: {:?}", src, dst, event_type);
        if dst != src {
            let json_body = serde_json::json!({
                "type": event_type,
                "data": {
                    "old_stub_indecies": [src],
                    "new_stub_indecies": [dst],
                }
            });
            json_body.to_string()
        } else {
            // only one stub
            let json_body = serde_json::json!({
                "type": event_type,
                "data": {
                    "stub_indecies": [src],
                }
            });
            json_body.to_string()
        }
    }
}
