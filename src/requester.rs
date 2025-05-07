use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicI32, Ordering},
        OnceLock,
    },
    time::{Duration, Instant},
};

use reqwest::Response;
use tokio::{
    fs::File,
    io::{AsyncWriteExt, BufWriter},
    spawn,
    sync::{broadcast, oneshot},
    task::{yield_now, JoinHandle},
    time::sleep,
};
use tracing::instrument;

use crate::{dataset::Dataset, distribution::Distribution, scale_event::ScaleEvent};

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

#[allow(dead_code)]
async fn request(endpoint: &str, json_body: String) -> Result<Response, reqwest::Error> {
    Ok(reqwest::Client::builder()
        .no_proxy()
        .build()?
        .post(endpoint)
        .body(json_body)
        .header("Content-Type", "application/json")
        .send()
        .await?)
}

#[allow(dead_code)]
async fn request_with_timeout(
    endpoint: &str,
    json_body: String,
    timeout: Duration,
) -> Result<Response, reqwest::Error> {
    Ok(reqwest::Client::builder()
        .no_proxy()
        .timeout(timeout)
        .build()?
        .post(endpoint)
        .body(json_body)
        .header("Content-Type", "application/json")
        .send()
        .await?)
}

async fn wait_all(response_receiver: flume::Receiver<JoinHandle<()>>) {
    while let Ok(handle) = response_receiver.recv_async().await {
        handle.await.unwrap();
    }
}

/// Send requests in the open loop way.
///
/// Note:
/// - Intervals between requests are randomly generated.
/// - Use [`spawn_request_loop_with_timestamp`] instead if you want to control the intervals.
///
/// Await on the returned handle to wait for the loop to finish.
pub fn spawn_request_loop(
    endpoint: String,
    endpoints: Option<Vec<String>>,
    dataset: Dataset,
    prefill_only: bool,
    truncate: Option<u64>,
    protocol: Box<dyn crate::protocols::Protocol + Send>,
    interval_generator: IntervalGenerator,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    mut stopped: oneshot::Receiver<()>,
) -> JoinHandle<Result<(), i32>> {
    static BASETIME: OnceLock<Instant> = OnceLock::new();
    BASETIME.get_or_init(|| Instant::now());

    fn get_timestamp() -> u64 {
        BASETIME.get().unwrap().elapsed().as_millis() as u64
    }

    let parse_response = protocol.parse_response();

    let (tx, rx) = flume::unbounded();
    let handle = spawn(async move {
        wait_all(rx).await;
        Ok(())
    });

    spawn(async move {
        let mut timestamp = get_timestamp();
        let mut count = 0u64;
        loop {
            count += 1;
            if stopped.try_recv().is_ok() {
                break;
            }

            let endpoint = match cfg!(feature = "bypass_router") {
                false => Some(endpoint.clone()),
                true => endpoints
                    .as_ref()
                    .and_then(|vec| vec.get(count as usize % vec.len()).cloned()),
            }
            .clone();
            assert!(endpoint.is_some());
            let (input_length, output_length) = dataset.next_request(prefill_only, truncate);
            let json_body = protocol.request_json_body(input_length, output_length);
            let response_sender = response_sender.clone();
            let request_handle = spawn(async move {
                let s_time = get_timestamp();
                if let Ok(response) = request_with_timeout(
                    endpoint.unwrap().as_str(),
                    json_body.to_string(),
                    Duration::from_secs(180.max((output_length as f64 * 0.4) as u64)),
                )
                .await
                {
                    let e_time = get_timestamp();
                    let mut metrics = parse_response(response);
                    metrics.insert("s_time".to_string(), s_time.to_string());
                    metrics.insert("e_time".to_string(), e_time.to_string());
                    metrics.insert("input_length".to_string(), input_length.to_string());
                    metrics.insert("output_length".to_string(), output_length.to_string());
                    response_sender.send(metrics).unwrap();
                } else {
                    tracing::error!(
                        "Request {} failed with input {} output {}",
                        count,
                        input_length,
                        output_length
                    );
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

#[instrument(skip_all)]
pub fn spawn_request_loop_with_timestamp(
    endpoint: String,
    endpoints: Option<Vec<String>>,
    dataset: Dataset,
    prefill_only: bool,
    truncate: Option<u64>,
    protocol: Box<dyn crate::protocols::Protocol + Send>,
    scale_factor: f64,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    scale_events: Option<ScaleEvent>,
    broadcast_tx: broadcast::Sender<()>,
) -> JoinHandle<Result<(), i32>> {
    static BASETIME: OnceLock<Instant> = OnceLock::new();
    static RETURNCODE: AtomicI32 = AtomicI32::new(0);
    BASETIME.get_or_init(|| Instant::now());
    fn get_timestamp() -> u64 {
        BASETIME.get().unwrap().elapsed().as_millis() as u64
    }

    let parse_response = protocol.parse_response();

    let rr = dataset.request_rate();
    println!("Origin request rate: {:.3} req/s", rr);
    println!("Scaled request rate: {:.3} req/s", rr * scale_factor);

    let (tx, rx) = flume::unbounded();
    let handle = spawn(async move {
        wait_all(rx).await;
        let a = RETURNCODE.load(Ordering::Relaxed);
        if a == 0 {
            Ok(())
        } else {
            Err(a)
        }
    });
    let mut scale_rx = broadcast_tx.subscribe();
    let mut gen_rx = broadcast_tx.subscribe();
    let new_endpoint = endpoint.clone();

    if scale_events.is_some() {
        spawn(async move {
            loop {
                if scale_rx.try_recv().is_ok() {
                    break;
                }
                let scale_endpoint = new_endpoint
                    .clone()
                    .split('/')
                    .take(3)
                    .collect::<Vec<&str>>()
                    .join("/")
                    + "/modify_cluster_state";
                let current_timestamp = get_timestamp();
                let (ts, src, dst, event_type) =
                    scale_events.as_ref().unwrap().next_event_with_timestamp();
                let next_timestamp = (ts as f64 / scale_factor) as u64;

                if next_timestamp > current_timestamp + 1 {
                    sleep(Duration::from_millis(next_timestamp - current_timestamp)).await;
                } else {
                    yield_now().await;
                }
                let json_body = scale_events
                    .as_ref()
                    .unwrap()
                    .request_json_body(src, dst, event_type);
                spawn(async move {
                    if let Err(_) = request_with_timeout(
                        scale_endpoint.as_str(),
                        json_body.to_string(),
                        Duration::from_secs(10),
                    )
                    .await
                    {
                        RETURNCODE.store(-1, Ordering::Relaxed);
                        tracing::error!(
                            "Scale request failed with src {} dst {} event_type {:?}",
                            src,
                            dst,
                            event_type
                        );
                    }
                });
            }
        });
    }
    spawn(async move {
        let mut count = 0u64;
        loop {
            count += 1;
            if gen_rx.try_recv().is_ok() {
                break;
            }

            // Get next request and its timestamp
            let endpoint = match cfg!(feature = "bypass_router") {
                false => Some(endpoint.clone()),
                true => endpoints
                    .as_ref()
                    .and_then(|vec| vec.get(count as usize % vec.len()).cloned()),
            };
            assert!(endpoint.is_some());

            let current_timestamp = get_timestamp();
            let (ts, input_length, output_length) =
                dataset.next_request_with_timestamp(prefill_only, truncate);
            let next_timestamp = (ts as f64 / scale_factor) as u64;

            if next_timestamp > current_timestamp + 1 {
                sleep(Duration::from_millis(next_timestamp - current_timestamp)).await;
            } else {
                yield_now().await;
            }

            let json_body = protocol.request_json_body(input_length, output_length);
            let response_sender = response_sender.clone();
            let request_handle = spawn(async move {
                let s_time = get_timestamp();
                if let Ok(response) = request_with_timeout(
                    endpoint.unwrap().as_str(),
                    json_body.to_string(),
                    Duration::from_secs(15 + output_length / 10),
                )
                .await
                {
                    let e_time = get_timestamp();
                    let mut metrics = parse_response(response);
                    metrics.insert("s_time".to_string(), s_time.to_string());
                    metrics.insert("e_time".to_string(), e_time.to_string());
                    metrics.insert("input_length".to_string(), input_length.to_string());
                    metrics.insert("output_length".to_string(), output_length.to_string());
                    response_sender.send(metrics).unwrap();
                } else {
                    RETURNCODE.store(-1, Ordering::Relaxed);
                    tracing::error!(
                        "Request {} failed with input {} output {}",
                        count,
                        input_length,
                        output_length
                    );
                }
            });

            tx.send_async(request_handle).await.unwrap();
        }
    });
    handle
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
        buf_writer.flush().await.unwrap();
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
