use std::{
    collections::BTreeMap,
    pin::Pin,
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc, OnceLock,
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

use crate::{
    apis::LLMApi, dataset::LLMTrace, distribution::Distribution,
    token_sampler::TokenSampler,
};

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
pub fn spawn_request_loop<A: 'static + LLMApi + Send>(
    endpoint: String,
    dataset: Arc<Pin<Box<dyn LLMTrace>>>,
    token_sampler: Arc<TokenSampler>,
    interval_generator: IntervalGenerator,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    mut stopped: oneshot::Receiver<()>,
) -> JoinHandle<Result<(), i32>> {
    static BASETIME: OnceLock<Instant> = OnceLock::new();
    BASETIME.get_or_init(|| Instant::now());

    fn get_timestamp() -> u64 {
        BASETIME.get().unwrap().elapsed().as_millis() as u64
    }

    let (tx, rx) = flume::unbounded();
    let handle = spawn(async move {
        wait_all(rx).await;
        Ok(())
    });

    spawn(async move {
        let mut timestamp = get_timestamp();
        let data_iter = dataset.iter();
        for data_index in data_iter {
            if stopped.try_recv().is_ok() {
                break;
            }
            // data to move into closure
            let endpoint = endpoint.clone();
            let response_sender = response_sender.clone();
            // TODO: add new span
            let (prompt, input_length, output_length) =
                dataset.inflate(data_index, token_sampler.as_ref());

            // parse in another coroutine
            let request_handle = spawn(async move {
                let json_body = A::request_json_body(prompt, output_length);
                let s_time = get_timestamp();
                if let Ok(response) = request_with_timeout(
                    endpoint.as_str(),
                    json_body.to_string(),
                    Duration::from_secs(180.max((output_length as f64 * 0.4) as u64)),
                )
                .await
                {
                    let e_time = get_timestamp();

                    let mut metrics = A::parse_response(response);
                    metrics.insert("s_time".to_string(), s_time.to_string());
                    metrics.insert("e_time".to_string(), e_time.to_string());
                    metrics.insert("input_length".to_string(), input_length.to_string());
                    metrics.insert("output_length".to_string(), output_length.to_string());

                    response_sender.send(metrics).unwrap();
                } else {
                    tracing::error!(
                        "Request {} timeout with input {} output {}",
                        data_index,
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

pub fn spawn_request_loop_with_timestamp<A: 'static + LLMApi + Send>(
    endpoint: String,
    dataset: Arc<Pin<Box<dyn LLMTrace>>>,
    token_sampler: Arc<TokenSampler>,
    scale_factor: f64,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    broadcast_tx: broadcast::Sender<()>,
) -> JoinHandle<Result<(), i32>> {
    static BASETIME: OnceLock<Instant> = OnceLock::new();
    static RETURNCODE: AtomicI32 = AtomicI32::new(0);
    BASETIME.get_or_init(|| Instant::now());
    fn get_timestamp() -> u64 {
        BASETIME.get().unwrap().elapsed().as_millis() as u64
    }

    let rr = dataset.rps();
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
    let mut gen_rx = broadcast_tx.subscribe();

    spawn(async move {
        let data_iter = dataset.iter();
        let endpoint = Arc::new(endpoint);
        for data_index in data_iter {
            if gen_rx.try_recv().is_ok() {
                break;
            }
            let endpoint = endpoint.clone();
            let response_sender = response_sender.clone();

            let curr_timestamp = get_timestamp();
            let next_timestamp = ((*dataset).timestamp(data_index) as f64 / scale_factor) as u64;

            if next_timestamp > curr_timestamp + 1 {
                sleep(Duration::from_millis(next_timestamp - curr_timestamp)).await;
            }

            // Do not parse in another coroutine to avoid sync/async lock contention 
            let (prompt, input_length, output_length) =
                dataset.inflate(data_index, token_sampler.as_ref());
            
            let request_handle = spawn(async move {
                let json_body = A::request_json_body(prompt, output_length);
                let s_time = get_timestamp();
                if let Ok(response) = request_with_timeout(
                    endpoint.as_str(),
                    json_body.to_string(),
                    Duration::from_secs(180.max((output_length as f64 * 0.4) as u64)),
                )
                .await
                {
                    let e_time = get_timestamp();

                    let mut metrics = A::parse_response(response);
                    metrics.insert("s_time".to_string(), s_time.to_string());
                    metrics.insert("e_time".to_string(), e_time.to_string());
                    metrics.insert("input_length".to_string(), input_length.to_string());
                    metrics.insert("output_length".to_string(), output_length.to_string());
                    metrics.insert("client_id".to_string(), data_index.to_string());

                    response_sender.send(metrics).unwrap();
                } else {
                    RETURNCODE.store(-1, Ordering::Release);
                    tracing::error!(
                        "Request {} failed with input {} output {}",
                        data_index,
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


pub fn spawn_request_loop_debug<A: 'static + LLMApi + Send>(
    _endpoint: String, // 保留参数，为了接口一致
    dataset: Arc<Pin<Box<dyn LLMTrace>>>,
    token_sampler: Arc<TokenSampler>,
    scale_factor: f64,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    broadcast_tx: broadcast::Sender<()>,
) -> JoinHandle<Result<(), i32>> {
    use std::time::Instant;
    static BASETIME: OnceLock<Instant> = OnceLock::new();
    static RETURNCODE: AtomicI32 = AtomicI32::new(0);
    BASETIME.get_or_init(|| Instant::now());

    fn get_timestamp() -> u64 {
        BASETIME.get().unwrap().elapsed().as_millis() as u64
    }

    let rr = dataset.rps();
    println!("Origin request rate: {:.3} req/s", rr);
    println!(
        "Scaled request rate (release-with-debug mode, no HTTP): {:.3} req/s",
        rr * scale_factor
    );

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

    let mut gen_rx = broadcast_tx.subscribe();
    let validate_tokenizer = Arc::new(token_sampler.get_tokenizer());

    spawn(async move {
        let data_iter = dataset.iter();
        for data_index in data_iter {
            if gen_rx.try_recv().is_ok() {
                break;
            }
            let tokenizer = validate_tokenizer.clone();
            let response_sender = response_sender.clone();

            let curr_timestamp = get_timestamp();
            // milisecond
            let next_timestamp = ((*dataset).timestamp(data_index) as f64 / scale_factor) as u64;

            if next_timestamp > curr_timestamp + 1 {
                sleep(Duration::from_millis(next_timestamp - curr_timestamp)).await;
            }

            let (sample, input_length, output_length) =
                dataset.inflate(data_index, token_sampler.as_ref());

            let request_handle = spawn(async move {
                let s_time = get_timestamp();
                let s_time_drift = s_time.saturating_sub(next_timestamp);

                let validate_len = tokenizer
                    .encode(sample.clone(), false)
                    .unwrap()
                    .get_ids()
                    .len();
                if validate_len != input_length as usize {
                    tracing::error!("Validation error: {input_length} :> {validate_len}");
                }
                
                let mut metrics = BTreeMap::new();
                metrics.insert("chat_id".to_string(), data_index.to_string());
                metrics.insert("input_length".to_string(), input_length.to_string());
                metrics.insert("output_length".to_string(), output_length.to_string());
                metrics.insert("s_time".to_string(), s_time.to_string());
                metrics.insert("s_time_drift".to_string(), s_time_drift.to_string());

                response_sender.send(metrics).unwrap();
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
    use crate::{
        dataset::{BailianDataset, LLMTrace},
        token_sampler::TokenSampler,
    };
    use tokenizers::Tokenizer;
    use std::sync::Arc;
    use tokio::fs::File;

    #[tokio::test]
    async fn test_inflate_latency() {
        // 初始化 tracing 输出
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_max_level(tracing::Level::DEBUG)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);

        // ====== 准备 dataset ======
        let mut dataset = BailianDataset::new();
        dataset.load("/Users/zdy/Workspace/Rust/request-sim/data/qwen-bailian-usagetraces-anon-main/qwen_traceA_blksz_16.jsonl"); // 你要准备一个小的测试文件

        let dataset = Arc::new(Box::pin(dataset) as Pin<Box<dyn LLMTrace>>);

        // ====== 准备 TokenSampler ======
        let token_sampler = Arc::new(TokenSampler::new(
            Tokenizer::from_file("/Users/zdy/Workspace/Rust/request-sim/data/tokenizer.json").unwrap(),
            "/Users/zdy/Workspace/Rust/request-sim/data/tokenizer_config.json".to_string(),
            4,     // num_producer
            128,   // capacity
            16,    // block size
        ));

        // ====== 准备输出通道 ======
        let (tx, rx) = flume::unbounded();
        let output_file = File::create("tmp/inflate_latency.jsonl").await.unwrap();
        let reporter = tokio::spawn(report_loop(output_file, rx));

        // ====== 测试循环 ======
        let iter = dataset.iter();
        for index in iter.take(10) { // 只测前10条
            let start = std::time::Instant::now();
            let (_prompt, input_len, output_len) = dataset.inflate(index, &token_sampler);
            let elapsed_us = start.elapsed().as_micros() as u64;

            let mut metrics = std::collections::BTreeMap::new();
            metrics.insert("index".to_string(), index.to_string());
            metrics.insert("input_length".to_string(), input_len.to_string());
            metrics.insert("output_length".to_string(), output_len.to_string());
            metrics.insert("inflate_time_us".to_string(), elapsed_us.to_string());
            tx.send_async(metrics).await.unwrap();
        }

        drop(tx);
        reporter.await.unwrap();

        tracing::info!("Inflate latency test completed");
    }
}
