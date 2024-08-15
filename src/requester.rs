use std::{
    collections::BTreeMap,
    sync::OnceLock,
    time::{Duration, Instant},
};

use reqwest::Response;
use tokio::{
    fs::File,
    io::{AsyncWriteExt, BufWriter},
    spawn,
    sync::oneshot,
    task::{yield_now, JoinHandle},
    time::{sleep, timeout},
};

use crate::{dataset::Dataset, distribution::Distribution, protocols::Protocol};

pub struct IntervalGenerator {
    distribution: Box<dyn Distribution>,
}

impl IntervalGenerator {
    pub fn new<D: Distribution + 'static>(distribution: D) -> Self {
        Self {
            distribution: Box::new(distribution),
        }
    }

    pub fn interval_in_millis(&self) -> f64 {
        self.distribution.generate()
    }
}

/// Create a new interval generator with a gamma distribution.
///
/// `request_rate`: how many requests per second
pub fn create_gamma_interval_generator(request_rate: f64, cv: f64) -> IntervalGenerator {
    let mean = 1000.0 / request_rate;
    let distribution = crate::distribution::gamma::Gamma::new(mean, cv);
    IntervalGenerator::new(distribution)
}

async fn request(endpoint: &str, json_body: String) -> Response {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(endpoint)
        .body(json_body)
        .header("Content-Type", "application/json")
        .send()
        .await
        .unwrap()
}

/// Send requests in the open loop way.
///
/// Note:
/// - Intervals between requests are randomly generated.
/// - Use [`request_loop_with_timestamp`] instead if you want to control the intervals.
///
/// Await on the returned handle to wait for the loop to finish.
pub fn spawn_request_loop<P: 'static + Protocol + Send>(
    endpoint: String,
    dataset: Dataset,
    protocol: P,
    interval_generator: IntervalGenerator,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    mut stopped: oneshot::Receiver<()>,
) -> JoinHandle<()> {
    async fn wait_all(response_receiver: flume::Receiver<JoinHandle<()>>) {
        while let Ok(handle) = response_receiver.recv_async().await {
            handle.await.unwrap();
        }
    }

    static BASETIME: OnceLock<Instant> = OnceLock::new();
    BASETIME.get_or_init(|| Instant::now());

    fn get_timestamp() -> u64 {
        BASETIME.get().unwrap().elapsed().as_millis() as u64
    }

    let (tx, rx) = flume::unbounded();
    let handle = spawn(wait_all(rx));

    spawn(async move {
        let mut timestamp = get_timestamp();
        loop {
            if stopped.try_recv().is_ok() {
                break;
            }
            let endpoint = endpoint.clone();
            let (input_length, output_length) = dataset.next_request();
            let json_body = protocol.request_json_body(input_length, output_length);
            let response_sender = response_sender.clone();
            let request_handle = spawn(async move {
                let s_time = get_timestamp();
                let slo = Duration::from_secs(output_length / 10 + 120);
                match timeout(slo, request(endpoint.as_str(), json_body.to_string())).await {
                    Ok(response) => {
                        let e_time = get_timestamp();

                        let mut metrics = P::parse_response(response);
                        metrics.insert("s_time".to_string(), s_time.to_string());
                        metrics.insert("e_time".to_string(), e_time.to_string());

                        if let Err(e) = response_sender.send(metrics) {
                            let error_file_path = "error.log";
                            let mut error_file = tokio::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(error_file_path)
                                .await
                                .expect("Failed to open error log file");
                            error_file
                                .write(format!("Error: {}\n", e).as_bytes())
                                .await
                                .expect("Failed to write to error log file");
                            error_file
                                .flush()
                                .await
                                .expect("Failed to flush error log file");
                        }
                    }
                    Err(_) => {
                        let error_file_path = "error.log";
                        eprintln!(
                            "Request with input {} output {} timeout!",
                            input_length, output_length
                        );
                        let mut error_file = tokio::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(error_file_path)
                            .await
                            .expect("Failed to open error log file");
                        error_file
                            .write(
                                format!(
                                    "Error: Request with input {} output {} timeout!\n",
                                    input_length, output_length
                                )
                                .as_bytes(),
                            )
                            .await
                            .expect("Failed to write to error log file");
                        error_file
                            .flush()
                            .await
                            .expect("Failed to flush error log file");
                    }
                }
            });

            tx.send_async(request_handle).await.unwrap();
            timestamp += interval_generator.interval_in_millis().round() as u64;
            let current_timestamp = get_timestamp();
            if timestamp > current_timestamp + 1 {
                sleep(Duration::from_millis(timestamp - current_timestamp)).await;
            } else {
                yield_now().await;
            }
        }
    });

    handle
}

pub async fn request_loop_with_timestamp<P: Protocol>() {
    unimplemented!("request_loop_with_timestamp");
}

/// The report loop writes the metrics to a file in JSONL format.
///
/// Report loop exits when the response receiver is closed.
pub async fn report_loop(
    mut output_jsonl_file: File,
    response_receiver: flume::Receiver<BTreeMap<String, String>>,
) {
    let mut buf_writer = BufWriter::new(&mut output_jsonl_file);
    while let Ok(metrics) = response_receiver.recv_async().await {
        let line = serde_json::to_string(&metrics).unwrap();
        buf_writer.write_all(line.as_bytes()).await.unwrap();
        buf_writer.write_all(b"\n").await.unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_json() {
        let mut map = BTreeMap::new();
        map.insert("a", "a");
        map.insert("b", "a");
        map.insert("c", "a");
        map.insert("d", "a");
        map.insert("e", "a");
        map.insert("f", "a");
        let line = serde_json::to_string(&map).unwrap();
        println!("{}", line);
    }
}
