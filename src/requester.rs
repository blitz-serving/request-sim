use std::{
    collections::{BTreeMap, HashMap},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use reqwest::Response;
use tokio::{
    fs::File,
    io::{AsyncWriteExt, BufWriter},
    spawn,
    task::JoinHandle,
    time::{sleep, sleep_until, Instant as TokioInstant},
};

use crate::apis::METRIC_PERCENTILES;
use crate::cache::PromptCache;
use crate::dataset::PromptPayload;
use crate::{
    apis::{InFlightState, LLMApi, RequestError},
    dataset::LLMTrace,
    get_timestamp, init_basetime, timeout_secs_upon_slo,
};
#[cfg(feature = "prompt-text-hashed")]
use crate::dataset::ConversationGraph;
#[cfg(feature = "prompt-text-hashed")]
use crate::token_sampler::TokenSampler;

// ── Module-level statics ─────────────────────────────────────────────────────

static RETURNCODE: AtomicI32 = AtomicI32::new(0);

// ── Request context (unifies feature-gated inflate) ─────────────────────────

#[derive(Clone)]
pub struct RequestContext {
    pub dataset: Arc<Pin<Box<dyn LLMTrace>>>,
    #[cfg(feature = "prompt-text-hashed")]
    pub token_sampler: Arc<TokenSampler>,
    pub prompt_cache: Option<Arc<PromptCache>>,
}

impl RequestContext {
    /// Inflate prompt for a dataset entry, using the cache if available,
    /// otherwise delegating to the dataset's inflate method.
    pub fn inflate(&self, index: usize) -> (PromptPayload, u64, u64) {
        if let Some(ref cache) = self.prompt_cache {
            let entry = cache.get(index);
            (entry.prompt.clone(), entry.input_length, entry.output_length)
        } else {
            #[cfg(feature = "prompt-text-hashed")]
            {
                self.dataset.inflate(index, self.token_sampler.as_ref())
            }
            #[cfg(feature = "prompt-text-plain")]
            {
                self.dataset.inflate(index)
            }
        }
    }
}

// ── Arrival process for random-process mode ─────────────────────────────────

// ── Output tracking for multi-turn KV cache consistency ─────────────────────

/// Shared state for tracking engine output across multi-turn conversations.
/// When enabled, the dispatch loop waits for ancestor turns to complete,
/// then constructs a Messages prompt with actual output as assistant messages.
#[cfg(feature = "prompt-text-hashed")]
#[derive(Clone)]
pub struct TrackOutputState {
    pub graph: Arc<ConversationGraph>,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    completions: Arc<dashmap::DashMap<usize, tokio::sync::watch::Sender<Option<String>>>>,
}

#[cfg(feature = "prompt-text-hashed")]
impl TrackOutputState {
    pub fn new(graph: ConversationGraph, tokenizer: tokenizers::Tokenizer, dataset_len: usize) -> Self {
        let completions = Arc::new(dashmap::DashMap::with_capacity(dataset_len));
        for i in 0..dataset_len {
            let (tx, _rx) = tokio::sync::watch::channel(None);
            completions.insert(i, tx);
        }
        Self {
            graph: Arc::new(graph),
            tokenizer: Arc::new(tokenizer),
            completions,
        }
    }

    /// Store the output text for a completed request and notify waiters.
    pub fn complete(&self, data_index: usize, output_text: String) {
        if let Some(tx) = self.completions.get(&data_index) {
            let _ = tx.send(Some(output_text));
        }
    }

    /// Get the stored output text for a data_index (non-blocking).
    pub fn get_output(&self, data_index: usize) -> Option<String> {
        self.completions
            .get(&data_index)
            .and_then(|tx| tx.borrow().clone())
    }

    /// Wait for ALL ancestor outputs to be available.
    pub async fn wait_for_ancestors(&self, data_index: usize) {
        let chain = self.graph.get_chain(data_index);
        for &ancestor_idx in &chain {
            let mut rx = {
                let Some(entry) = self.completions.get(&ancestor_idx) else {
                    continue;
                };
                entry.subscribe()
            };
            loop {
                if rx.borrow().is_some() {
                    break;
                }
                if rx.changed().await.is_err() {
                    break;
                }
            }
        }
    }

    /// Get the ancestor chain and their captured outputs.
    /// Returns (chain, outputs) where chain = [root, ..., parent] and outputs[i] = output of chain[i].
    pub fn get_chain_with_outputs(&self, data_index: usize) -> (Vec<usize>, Vec<String>) {
        let chain = self.graph.get_chain(data_index);
        let outputs: Vec<String> = chain
            .iter()
            .filter_map(|&idx| self.get_output(idx))
            .collect();
        (chain, outputs)
    }
}

// ── Arrival process (continued) ─────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum ArrivalProcess {
    Poisson,
    Uniform,
}

fn next_interarrival(arrival: &ArrivalProcess, rate: f64) -> Duration {
    use rand::Rng;
    match arrival {
        ArrivalProcess::Poisson => {
            let mut rng = rand::thread_rng();
            let exp = rand_distr::Exp::new(rate).unwrap();
            Duration::from_secs_f64(rng.sample(exp))
        }
        ArrivalProcess::Uniform => Duration::from_secs_f64(1.0 / rate),
    }
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

async fn post_with_timeout<A: 'static + LLMApi + Send>(
    client: reqwest::Client,
    endpoint: &str,
    json_body: String,
    timeout: Duration,
    stream: bool,
    in_flight: Option<Arc<InFlightState>>,
) -> Result<BTreeMap<String, String>, RequestError> {
    let mut req = client
        .post(endpoint)
        .body(json_body.clone())
        .header("Content-Type", "application/json");

    if !stream {
        req = req.timeout(timeout);
    }
    for (name, value) in A::extra_headers(&json_body) {
        req = req.header(name, value);
    }

    let response = req.send().await.map_err(|e| RequestError::Other(e))?;

    A::parse_response(response, stream, timeout, in_flight).await
}

async fn wait_all(handle_rx: flume::Receiver<JoinHandle<()>>, interrupt_flag: Arc<AtomicBool>) {
    while let Ok(handle) = handle_rx.recv_async().await {
        handle.await.unwrap();
        if interrupt_flag.load(Ordering::Relaxed) {
            tracing::info!("{} requests has not yet finished!", handle_rx.len());
        }
    }
}

#[cfg(feature = "prompt-text-hashed")]
pub fn spawn_request_loop_with_timestamp<A: 'static + LLMApi + Send>(
    endpoint: String,
    dataset: Arc<Pin<Box<dyn LLMTrace>>>,
    token_sampler: Arc<TokenSampler>,
    scale_factor: f64,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    interrupt_flag: Arc<AtomicBool>,
    ttft_slo: f32,
    tpot_slo: f32,
    stream: bool,
    early_stop_error_threshold: Option<u32>,
    prompt_cache: Option<Arc<PromptCache>>,
    time_range: (Option<u64>, Option<u64>),
    track_output: Option<TrackOutputState>,
) -> JoinHandle<Result<(), i32>> {
    init_basetime();

    let rr = dataset.rps();
    println!("Origin request rate: {:.3} req/s", rr);
    println!("Scaled request rate: {:.3} req/s", rr * scale_factor);

    let (tx, rx) = flume::unbounded();
    let flag = Arc::clone(&interrupt_flag);
    let handle = spawn(async move {
        wait_all(rx, flag).await;
        let a = RETURNCODE.load(Ordering::Relaxed);
        if a == 0 {
            Ok(())
        } else {
            Err(a)
        }
    });

    let error_count = Arc::new(AtomicU32::new(0));

    spawn(async move {
        let data_iter = dataset.iter();
        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .unwrap();
        let endpoint = Arc::new(endpoint);
        let (begin_time, end_time) = time_range;
        for data_index in data_iter {
            let trace_ts = dataset.timestamp(data_index);
            if let Some(begin) = begin_time {
                if trace_ts < begin {
                    continue;
                }
            }
            if let Some(end) = end_time {
                if trace_ts > end {
                    continue;
                }
            }

            let error_count = Arc::clone(&error_count);
            if interrupt_flag.load(Ordering::Relaxed) {
                break;
            }

            if let Some(threshold) = early_stop_error_threshold {
                if threshold <= error_count.load(Ordering::Relaxed) {
                    tracing::error!(
                        "Request error accumulated more than threshold: {}, exit client",
                        threshold
                    );
                    interrupt_flag.store(true, Ordering::SeqCst);
                    break;
                }
            }
            let client = http_client.clone();
            let endpoint = endpoint.clone();
            let response_sender = response_sender.clone();

            // Output tracking: wait for all ancestor outputs, then inflate as Messages
            let use_messages = if let Some(ref tos) = track_output {
                if tos.graph.parent_index[data_index].is_some() {
                    tos.wait_for_ancestors(data_index).await;
                    true
                } else {
                    false
                }
            } else {
                false
            };

            let curr_timestamp = get_timestamp() as u64;
            let next_timestamp = ((*dataset).timestamp(data_index) as f64 / scale_factor) as u64;

            if next_timestamp > curr_timestamp + 1 {
                sleep(Duration::from_millis(next_timestamp - curr_timestamp)).await;
            }

            // Inflate: use Messages for multi-turn with tracked output, Content otherwise
            let (prompt, input_length, output_length) = if use_messages {
                let tos = track_output.as_ref().unwrap();
                let (chain, outputs) = tos.get_chain_with_outputs(data_index);
                dataset
                    .inflate_as_messages(
                        data_index,
                        token_sampler.as_ref(),
                        &tos.graph,
                        &chain,
                        &outputs,
                    )
                    .unwrap_or_else(|| dataset.inflate(data_index, token_sampler.as_ref()))
            } else if let Some(ref cache) = prompt_cache {
                let entry = cache.get(data_index);
                (entry.prompt.clone(), entry.input_length, entry.output_length)
            } else {
                dataset.inflate(data_index, token_sampler.as_ref())
            };

            let track_output_clone = track_output.clone();
            let request_handle = spawn(async move {
                let json_body = A::request_json_body(prompt, input_length, output_length, stream);
                let s_time = get_timestamp();
                let s_time_drift = s_time - next_timestamp as f64;
                match post_with_timeout::<A>(
                    client,
                    endpoint.as_str(),
                    json_body.to_string(),
                    Duration::from_secs(timeout_secs_upon_slo(output_length, ttft_slo, tpot_slo)),
                    stream,
                    None,
                )
                .await
                {
                    Ok(mut metrics) => {
                        let e_time = get_timestamp();

                        metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                        metrics.insert("s_time_drift".to_string(), format!("{s_time_drift:.3}"));
                        metrics.insert("e_time".to_string(), format!("{e_time:.3}"));
                        metrics.insert("input_length".to_string(), input_length.to_string());
                        metrics.insert("output_length".to_string(), output_length.to_string());

                        let span_time = e_time - s_time;
                        metrics.insert(
                            "span_time".to_string(),
                            format!("{span_time:.3}"),
                        );

                        if output_length > 0
                            && metrics
                                .get("status")
                                .map(|s| s.starts_with('2'))
                                .unwrap_or(false)
                        {
                            let normalized_e2e = span_time / output_length as f64;
                            metrics.insert("normalized_e2e".to_string(), format!("{normalized_e2e:.3}"));
                        }

                        // Store output text for output tracking
                        if let Some(ref tos) = track_output_clone {
                            if let Some(text) = metrics.remove("output_text") {
                                tos.complete(data_index, text);
                            }
                        }

                        response_sender.send(metrics).unwrap();
                    }
                    Err(RequestError::Timeout) => {
                        let e_time = get_timestamp();

                        let mut metrics = BTreeMap::<String, String>::from([(
                            "status".to_owned(),
                            "timeout".to_owned(),
                        )]);
                        metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                        metrics.insert("s_time_drift".to_string(), format!("{s_time_drift:.3}"));
                        metrics.insert("e_time".to_string(), format!("{e_time:.3}"));
                        metrics.insert("input_length".to_string(), input_length.to_string());
                        metrics.insert("output_length".to_string(), output_length.to_string());

                        let span_time = e_time - s_time;
                        metrics.insert(
                            "span_time".to_string(),
                            format!("{span_time:.3}"),
                        );
                        response_sender.send(metrics).unwrap();
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RequestError::Other(error)) => {
                        tracing::error!(
                            "Request#{data_index}::({input_length}|{output_length}) error: {error}",
                        );
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RequestError::StreamErr(error)) => {
                        tracing::error!(
                            "Request#{data_index}::({input_length}|{output_length}) stream error: {error}",
                        );
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            tx.send_async(request_handle).await.unwrap();
        }
        tracing::debug!("Requester exited.");
    });
    handle
}

#[cfg(feature = "prompt-text-plain")]
pub fn spawn_request_loop_with_timestamp<A: 'static + LLMApi + Send>(
    endpoint: String,
    dataset: Arc<Pin<Box<dyn LLMTrace>>>,
    scale_factor: f64,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    interrupt_flag: Arc<AtomicBool>,
    ttft_slo: f32,
    tpot_slo: f32,
    stream: bool,
    early_stop_error_threshold: Option<u32>,
    prompt_cache: Option<Arc<PromptCache>>,
    time_range: (Option<u64>, Option<u64>),
) -> JoinHandle<Result<(), i32>> {
    init_basetime();

    let rr = dataset.rps();
    println!("Origin request rate: {:.3} req/s", rr);
    println!("Scaled request rate: {:.3} req/s", rr * scale_factor);

    let (tx, rx) = flume::unbounded();
    let flag = Arc::clone(&interrupt_flag);
    let handle = spawn(async move {
        wait_all(rx, flag).await;
        let a = RETURNCODE.load(Ordering::Relaxed);
        if a == 0 {
            Ok(())
        } else {
            Err(a)
        }
    });

    let error_count = Arc::new(AtomicU32::new(0));

    spawn(async move {
        let data_iter = dataset.iter();
        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .unwrap();
        let endpoint = Arc::new(endpoint);
        let (begin_time, end_time) = time_range;
        for data_index in data_iter {
            let trace_ts = dataset.timestamp(data_index);
            if let Some(begin) = begin_time {
                if trace_ts < begin {
                    continue;
                }
            }
            if let Some(end) = end_time {
                if trace_ts > end {
                    continue;
                }
            }

            let error_count = Arc::clone(&error_count);
            if interrupt_flag.load(Ordering::Relaxed) {
                break;
            }

            if let Some(threshold) = early_stop_error_threshold {
                if threshold <= error_count.load(Ordering::Relaxed) {
                    tracing::error!(
                        "Request error accumulated more than threshold: {}, exit client",
                        threshold
                    );
                    interrupt_flag.store(true, Ordering::SeqCst);
                    break;
                }
            }
            let client = http_client.clone();
            let endpoint = endpoint.clone();
            let response_sender = response_sender.clone();

            let curr_timestamp = get_timestamp() as u64;
            let next_timestamp = ((*dataset).timestamp(data_index) as f64 / scale_factor) as u64;

            if next_timestamp > curr_timestamp + 1 {
                sleep(Duration::from_millis(next_timestamp - curr_timestamp)).await;
            }

            let (prompt, input_length, output_length) =
                if let Some(ref cache) = prompt_cache {
                    let entry = cache.get(data_index);
                    (entry.prompt.clone(), entry.input_length, entry.output_length)
                } else {
                    dataset.inflate(data_index)
                };

            let request_handle = spawn(async move {
                let json_body = A::request_json_body(prompt, input_length, output_length, stream);
                let s_time = get_timestamp();
                let s_time_drift = s_time - next_timestamp as f64;
                match post_with_timeout::<A>(
                    client,
                    endpoint.as_str(),
                    json_body.to_string(),
                    Duration::from_secs(timeout_secs_upon_slo(output_length, ttft_slo, tpot_slo)),
                    stream,
                    None,
                )
                .await
                {
                    Ok(mut metrics) => {
                        let e_time = get_timestamp();

                        metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                        metrics.insert("s_time_drift".to_string(), format!("{s_time_drift:.3}"));
                        metrics.insert("e_time".to_string(), format!("{e_time:.3}"));
                        metrics.insert("input_length".to_string(), input_length.to_string());
                        metrics.insert("output_length".to_string(), output_length.to_string());

                        let span_time = e_time - s_time;
                        metrics.insert(
                            "span_time".to_string(),
                            format!("{span_time:.3}"),
                        );

                        if output_length > 0
                            && metrics
                                .get("status")
                                .map(|s| s.starts_with('2'))
                                .unwrap_or(false)
                        {
                            let normalized_e2e = span_time / output_length as f64;
                            metrics.insert("normalized_e2e".to_string(), format!("{normalized_e2e:.3}"));
                        }

                        response_sender.send(metrics).unwrap();
                    }
                    Err(RequestError::Timeout) => {
                        let e_time = get_timestamp();

                        let mut metrics = BTreeMap::<String, String>::from([(
                            "status".to_owned(),
                            "timeout".to_owned(),
                        )]);
                        metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                        metrics.insert("s_time_drift".to_string(), format!("{s_time_drift:.3}"));
                        metrics.insert("e_time".to_string(), format!("{e_time:.3}"));
                        metrics.insert("input_length".to_string(), input_length.to_string());
                        metrics.insert("output_length".to_string(), output_length.to_string());

                        let span_time = e_time - s_time;
                        metrics.insert(
                            "span_time".to_string(),
                            format!("{span_time:.3}"),
                        );
                        response_sender.send(metrics).unwrap();
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RequestError::Other(error)) => {
                        tracing::error!(
                            "Request#{data_index}::({input_length}|{output_length}) error: {error}",
                        );
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RequestError::StreamErr(error)) => {
                        tracing::error!(
                            "Request#{data_index}::({input_length}|{output_length}) stream error: {error}",
                        );
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            tx.send_async(request_handle).await.unwrap();
        }
        tracing::debug!("Requester exited.");
    });
    handle
}

#[cfg(feature = "prompt-text-hashed")]
pub fn spawn_request_loop_debug<A: 'static + LLMApi + Send>(
    _endpoint: String,
    dataset: Arc<Pin<Box<dyn LLMTrace>>>,
    token_sampler: Arc<TokenSampler>,
    scale_factor: f64,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    interrupt_flag: Arc<AtomicBool>,
    prompt_cache: Option<Arc<PromptCache>>,
    time_range: (Option<u64>, Option<u64>),
) -> JoinHandle<Result<(), i32>> {
    init_basetime();

    let rr = dataset.rps();
    println!("Origin request rate: {:.3} req/s", rr);
    println!(
        "Scaled request rate (release-with-debug mode, no HTTP): {:.3} req/s",
        rr * scale_factor
    );

    let (tx, rx) = flume::unbounded();
    let flag = Arc::clone(&interrupt_flag);
    let handle = spawn(async move {
        wait_all(rx, flag).await;
        let a = RETURNCODE.load(Ordering::Relaxed);
        if a == 0 {
            Ok(())
        } else {
            Err(a)
        }
    });

    let validate_tokenizer = Arc::new(token_sampler.get_tokenizer());

    spawn(async move {
        let data_iter = dataset.iter();
        let (begin_time, end_time) = time_range;
        for data_index in data_iter {
            let trace_ts = dataset.timestamp(data_index);
            if let Some(begin) = begin_time {
                if trace_ts < begin {
                    continue;
                }
            }
            if let Some(end) = end_time {
                if trace_ts > end {
                    continue;
                }
            }

            if interrupt_flag.load(Ordering::Relaxed) {
                break;
            }
            let tokenizer = validate_tokenizer.clone();
            let response_sender = response_sender.clone();

            let curr_timestamp = get_timestamp() as u64;
            let next_timestamp = ((*dataset).timestamp(data_index) as f64 / scale_factor) as u64;

            if next_timestamp > curr_timestamp + 1 {
                sleep(Duration::from_millis(next_timestamp - curr_timestamp)).await;
            }

            let (payload, input_length, output_length) =
                if let Some(ref cache) = prompt_cache {
                    let entry = cache.get(data_index);
                    (entry.prompt.clone(), entry.input_length, entry.output_length)
                } else {
                    dataset.inflate(data_index, token_sampler.as_ref())
                };
            let sample = match payload {
                PromptPayload::Content(s) => s,
                _ => panic!("debug mode requires Content prompt (hashed dataset)"),
            };

            let request_handle = spawn(async move {
                let s_time = get_timestamp();
                let s_time_drift = s_time - next_timestamp as f64;

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
                metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                metrics.insert("s_time_drift".to_string(), format!("{s_time_drift:.3}"));

                response_sender.send(metrics).unwrap();
            });

            tx.send_async(request_handle).await.unwrap();
        }
        tracing::debug!("Requester exited.");
    });

    handle
}

#[cfg(feature = "prompt-text-plain")]
pub fn spawn_request_loop_debug<A: 'static + LLMApi + Send>(
    _endpoint: String,
    dataset: Arc<Pin<Box<dyn LLMTrace>>>,
    scale_factor: f64,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    interrupt_flag: Arc<AtomicBool>,
    prompt_cache: Option<Arc<PromptCache>>,
    time_range: (Option<u64>, Option<u64>),
) -> JoinHandle<Result<(), i32>> {
    init_basetime();

    let rr = dataset.rps();
    println!("Origin request rate: {:.3} req/s", rr);
    println!(
        "Scaled request rate (release-with-debug mode, no HTTP): {:.3} req/s",
        rr * scale_factor
    );

    let (tx, rx) = flume::unbounded();
    let flag = Arc::clone(&interrupt_flag);
    let handle = spawn(async move {
        wait_all(rx, flag).await;
        let a = RETURNCODE.load(Ordering::Relaxed);
        if a == 0 {
            Ok(())
        } else {
            Err(a)
        }
    });

    spawn(async move {
        let data_iter = dataset.iter();
        let (begin_time, end_time) = time_range;
        for data_index in data_iter {
            let trace_ts = dataset.timestamp(data_index);
            if let Some(begin) = begin_time {
                if trace_ts < begin {
                    continue;
                }
            }
            if let Some(end) = end_time {
                if trace_ts > end {
                    continue;
                }
            }

            if interrupt_flag.load(Ordering::Relaxed) {
                break;
            }
            let response_sender = response_sender.clone();

            let curr_timestamp = get_timestamp() as u64;
            let next_timestamp = ((*dataset).timestamp(data_index) as f64 / scale_factor) as u64;

            if next_timestamp > curr_timestamp + 1 {
                sleep(Duration::from_millis(next_timestamp - curr_timestamp)).await;
            }

            let (sample, input_length, output_length) =
                if let Some(ref cache) = prompt_cache {
                    let entry = cache.get(data_index);
                    (entry.prompt.clone(), entry.input_length, entry.output_length)
                } else {
                    dataset.inflate(data_index)
                };

            let request_handle = spawn(async move {
                let s_time = get_timestamp();
                let s_time_drift = s_time - next_timestamp as f64;

                let _ = &sample; // used for potential future validation

                let mut metrics = BTreeMap::new();
                metrics.insert("chat_id".to_string(), data_index.to_string());
                metrics.insert("input_length".to_string(), input_length.to_string());
                metrics.insert("output_length".to_string(), output_length.to_string());
                metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                metrics.insert("s_time_drift".to_string(), format!("{s_time_drift:.3}"));

                response_sender.send(metrics).unwrap();
            });

            tx.send_async(request_handle).await.unwrap();
        }
        tracing::debug!("Requester exited.");
    });

    handle
}

// ── Random-process mode ─────────────────────────────────────────────────────

pub fn spawn_request_loop_random_process<A: 'static + LLMApi + Send>(
    endpoint: String,
    ctx: RequestContext,
    arrival: ArrivalProcess,
    rate: f64,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    interrupt_flag: Arc<AtomicBool>,
    ttft_slo: f32,
    tpot_slo: f32,
    stream: bool,
    early_stop_error_threshold: Option<u32>,
) -> JoinHandle<Result<(), i32>> {
    init_basetime();

    println!("Random-process mode: arrival={:?}, rate={:.3} req/s", arrival, rate);

    let (tx, rx) = flume::unbounded();
    let flag = Arc::clone(&interrupt_flag);
    let handle = spawn(async move {
        wait_all(rx, flag).await;
        let a = RETURNCODE.load(Ordering::Relaxed);
        if a == 0 { Ok(()) } else { Err(a) }
    });

    let error_count = Arc::new(AtomicU32::new(0));

    spawn(async move {
        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .unwrap();
        let endpoint = Arc::new(endpoint);
        let dataset_len = ctx.dataset.len();
        let mut index: usize = 0;

        // Deadline-based scheduling: anchor to absolute time to prevent drift.
        // For uniform arrival, each request i dispatches at loop_start + i/rate,
        // so inflate() and spawn overhead do not accumulate into the interval.
        let loop_start = TokioInstant::now();
        let loop_start_ms = get_timestamp();

        while !interrupt_flag.load(Ordering::Relaxed) {
            let data_index = index % dataset_len;

            let error_count = Arc::clone(&error_count);
            if let Some(threshold) = early_stop_error_threshold {
                if threshold <= error_count.load(Ordering::Relaxed) {
                    tracing::error!(
                        "Request error accumulated more than threshold: {}, exit client",
                        threshold
                    );
                    interrupt_flag.store(true, Ordering::SeqCst);
                    break;
                }
            }

            // Inter-arrival delay: deadline-based for uniform, sleep-based for Poisson
            let intended_ms = match &arrival {
                ArrivalProcess::Uniform => {
                    // Absolute deadline: loop_start + index/rate. Index 0 dispatches immediately.
                    let deadline = loop_start + Duration::from_secs_f64(index as f64 / rate);
                    sleep_until(deadline).await;
                    loop_start_ms + index as f64 * 1000.0 / rate
                }
                ArrivalProcess::Poisson => {
                    sleep(next_interarrival(&arrival, rate)).await;
                    get_timestamp()
                }
            };

            index += 1;

            let client = http_client.clone();
            let endpoint = endpoint.clone();
            let response_sender = response_sender.clone();

            let (prompt, input_length, output_length) = ctx.inflate(data_index);

            let request_handle = spawn(async move {
                let json_body = A::request_json_body(prompt, input_length, output_length, stream);
                let s_time = get_timestamp();
                let s_time_drift = s_time - intended_ms;
                match post_with_timeout::<A>(
                    client,
                    endpoint.as_str(),
                    json_body.to_string(),
                    Duration::from_secs(timeout_secs_upon_slo(output_length, ttft_slo, tpot_slo)),
                    stream,
                    None,
                )
                .await
                {
                    Ok(mut metrics) => {
                        let e_time = get_timestamp();
                        metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                        metrics.insert("s_time_drift".to_string(), format!("{s_time_drift:.3}"));
                        metrics.insert("e_time".to_string(), format!("{e_time:.3}"));
                        metrics.insert("input_length".to_string(), input_length.to_string());
                        metrics.insert("output_length".to_string(), output_length.to_string());
                        let span_time = e_time - s_time;
                        metrics.insert("span_time".to_string(), format!("{span_time:.3}"));
                        if output_length > 0
                            && metrics.get("status").map(|s| s.starts_with('2')).unwrap_or(false)
                        {
                            let normalized_e2e = span_time / output_length as f64;
                            metrics.insert("normalized_e2e".to_string(), format!("{normalized_e2e:.3}"));
                        }
                        response_sender.send(metrics).unwrap();
                    }
                    Err(RequestError::Timeout) => {
                        let e_time = get_timestamp();
                        let mut metrics = BTreeMap::<String, String>::from([(
                            "status".to_owned(), "timeout".to_owned(),
                        )]);
                        metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                        metrics.insert("s_time_drift".to_string(), format!("{s_time_drift:.3}"));
                        metrics.insert("e_time".to_string(), format!("{e_time:.3}"));
                        metrics.insert("input_length".to_string(), input_length.to_string());
                        metrics.insert("output_length".to_string(), output_length.to_string());
                        let span_time = e_time - s_time;
                        metrics.insert("span_time".to_string(), format!("{span_time:.3}"));
                        response_sender.send(metrics).unwrap();
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RequestError::Other(error)) => {
                        tracing::error!(
                            "Request#{data_index}::({input_length}|{output_length}) error: {error}",
                        );
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RequestError::StreamErr(error)) => {
                        tracing::error!(
                            "Request#{data_index}::({input_length}|{output_length}) stream error: {error}",
                        );
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            tx.send_async(request_handle).await.unwrap();
        }
        tracing::debug!("Random-process requester exited.");
    });
    handle
}

// ── Feedback mode (control-theory closed-loop) ──────────────────────────────

// ── AIMD Feedback Controller ─────────────────────────────────────────────────

/// Configuration for the AIMD feedback controller.
/// When all constraint fields are None, feedback mode uses the static semaphore (backward compat).
pub struct ControllerConfig {
    pub bs_limit: usize,
    pub interval_secs: f64,
    pub cooldown_ticks: u32,
    pub tpot_limit_ms: Option<f64>,
    pub tps_limit: Option<f64>,
    pub all_tokens_limit: Option<u64>,
}

/// Shared mutable state between the dispatch loop, per-request tasks, and the controller.
struct FeedbackState {
    /// Currently in-flight requests, keyed by request ID.
    in_flight: Mutex<HashMap<u64, Arc<InFlightState>>>,
    /// Monotonically increasing total output tokens received across all requests.
    global_token_counter: AtomicU64,
    /// Current allowed concurrency (controller writes, dispatch reads).
    bs_allowed: AtomicUsize,
    /// Actual number of active (in-flight) requests right now.
    bs_active: AtomicUsize,
    /// Wakes the dispatch loop when a request completes or bs_allowed increases.
    notify: tokio::sync::Notify,
    /// Monotonic request ID generator.
    next_request_id: AtomicU64,
}

impl FeedbackState {
    fn new(initial_bs: usize) -> Self {
        Self {
            in_flight: Mutex::new(HashMap::new()),
            global_token_counter: AtomicU64::new(0),
            bs_allowed: AtomicUsize::new(initial_bs),
            bs_active: AtomicUsize::new(0),
            notify: tokio::sync::Notify::new(),
            next_request_id: AtomicU64::new(0),
        }
    }
}

/// Spawn the AIMD controller loop as a background tokio task.
///
/// Each tick (every `interval_secs`):
/// 1. Observe: bs_active, tps, avg_tpot, all_tokens
/// 2. Evaluate constraints (tpot_limit, tps_limit, all_tokens_limit)
/// 3. AIMD on control input (step_size): AI grows step on success, hard-reset on violation
/// 4. Actuate: adjust bs_allowed toward bs_limit
fn spawn_controller_loop(
    config: &ControllerConfig,
    state: Arc<FeedbackState>,
    interrupt_flag: Arc<AtomicBool>,
) -> JoinHandle<()> {
    let bs_limit = config.bs_limit;
    let interval_secs = config.interval_secs;
    let cooldown_ticks = config.cooldown_ticks;
    let tpot_limit_ms = config.tpot_limit_ms;
    let tps_limit = config.tps_limit;
    let all_tokens_limit = config.all_tokens_limit;

    spawn(async move {
        let interval = Duration::from_secs_f64(interval_secs);
        let mut step_size: usize = 1;
        let mut cooldown_remaining: u32 = 0;

        loop {
            sleep(interval).await;

            if interrupt_flag.load(Ordering::Relaxed) {
                break;
            }

            // ── 1. Observe ──────────────────────────────────────────────
            let bs_active = state.bs_active.load(Ordering::Relaxed);

            // Compute per-user TPS (arithmetic mean of 1000/TPOT_i), avg TPOT,
            // and all_tokens from in-flight snapshot
            let (avg_tps, avg_tpot, all_tokens) = {
                let map = state.in_flight.lock().unwrap();
                let now_ms = get_timestamp();
                let mut tps_sum = 0.0;
                let mut tpot_sum = 0.0;
                let mut tpot_count = 0u64;
                let mut total_tokens: u64 = 0;

                for (_id, ifl) in map.iter() {
                    let out = ifl.output_tokens_so_far.load(Ordering::Relaxed);
                    total_tokens += ifl.input_length + out;

                    let ftt = ifl.first_token_time_ms.load(Ordering::Acquire);
                    if ftt > 0 && out > 1 {
                        let decode_elapsed = now_ms - ftt as f64;
                        if decode_elapsed > 0.0 {
                            let tpot_i = decode_elapsed / (out - 1) as f64;
                            tpot_sum += tpot_i;
                            tps_sum += 1000.0 / tpot_i;
                            tpot_count += 1;
                        }
                    }
                }
                let avg_tpot = if tpot_count > 0 {
                    tpot_sum / tpot_count as f64
                } else {
                    0.0
                };
                let avg_tps = if tpot_count > 0 {
                    tps_sum / tpot_count as f64
                } else {
                    0.0
                };
                (avg_tps, avg_tpot, total_tokens)
            };

            // ── 2. Evaluate constraints ─────────────────────────────────
            let tpot_violated = tpot_limit_ms
                .map(|limit| avg_tpot > limit && avg_tpot > 0.0)
                .unwrap_or(false);
            let tps_violated = tps_limit
                .map(|limit| avg_tps < limit && avg_tps > 0.0)
                .unwrap_or(false);
            let all_tokens_violated = all_tokens_limit
                .map(|limit| all_tokens > limit)
                .unwrap_or(false);
            let any_violated = tpot_violated || tps_violated || all_tokens_violated;

            // ── 3. AIMD on control input ────────────────────────────────
            let bs_current = state.bs_allowed.load(Ordering::Relaxed);

            let (new_bs, action) = if cooldown_remaining > 0 {
                cooldown_remaining -= 1;
                (bs_current, "cooldown")
            } else if any_violated {
                // Retreat: undo last advance, hard-reset step to 1
                let retreat = std::cmp::max(1, step_size - 1);
                let retreated = std::cmp::max(1, bs_current.saturating_sub(retreat));
                step_size = 1;
                cooldown_remaining = cooldown_ticks;
                (retreated, "retreat")
            } else if bs_current < bs_limit {
                // Advance: increase bs_allowed, grow step for next tick
                let new = std::cmp::min(bs_current + step_size, bs_limit);
                step_size += 1;
                cooldown_remaining = cooldown_ticks;
                (new, "advance")
            } else {
                (bs_current, "steady")
            };

            // ── 4. Actuate ──────────────────────────────────────────────
            let prev_bs = state.bs_allowed.swap(new_bs, Ordering::Release);
            if new_bs > prev_bs {
                state.notify.notify_one();
            }

            tracing::info!(
                "controller: bs_allowed={new_bs} bs_active={bs_active} tps={avg_tps:.1} \
                 tpot={avg_tpot:.1}ms all_tokens={all_tokens} step={step_size} action={action}"
            );
        }
    })
}

pub fn spawn_request_loop_feedback<A: 'static + LLMApi + Send>(
    endpoint: String,
    ctx: RequestContext,
    config: ControllerConfig,
    response_sender: flume::Sender<BTreeMap<String, String>>,
    interrupt_flag: Arc<AtomicBool>,
    ttft_slo: f32,
    tpot_slo: f32,
    stream: bool,
    early_stop_error_threshold: Option<u32>,
) -> JoinHandle<Result<(), i32>> {
    init_basetime();

    let bs_limit = config.bs_limit;
    println!(
        "Feedback mode: bs_limit={}, interval={}ms, dataset_size={}",
        bs_limit,
        (config.interval_secs * 1000.0) as u64,
        ctx.dataset.len()
    );

    let (tx, rx) = flume::unbounded();
    let flag = Arc::clone(&interrupt_flag);
    let handle = spawn(async move {
        wait_all(rx, flag).await;
        let a = RETURNCODE.load(Ordering::Relaxed);
        if a == 0 { Ok(()) } else { Err(a) }
    });

    let error_count = Arc::new(AtomicU32::new(0));
    let ctrl_interrupt = Arc::clone(&interrupt_flag);

    spawn(async move {
        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .unwrap();
        let endpoint = Arc::new(endpoint);

        let state = Arc::new(FeedbackState::new(1));
        let _controller_handle =
            spawn_controller_loop(&config, Arc::clone(&state), ctrl_interrupt);

        let data_iter = ctx.dataset.iter();

        for data_index in data_iter {
            if interrupt_flag.load(Ordering::Relaxed) {
                break;
            }

            let error_count = Arc::clone(&error_count);
            if let Some(threshold) = early_stop_error_threshold {
                if threshold <= error_count.load(Ordering::Relaxed) {
                    tracing::error!(
                        "Request error accumulated more than threshold: {}, exit client",
                        threshold
                    );
                    interrupt_flag.store(true, Ordering::SeqCst);
                    break;
                }
            }

            // Concurrency gate: wait until bs_active < bs_allowed
            loop {
                let notified = state.notify.notified();
                if state.bs_active.load(Ordering::Relaxed)
                    < state.bs_allowed.load(Ordering::Acquire)
                {
                    break;
                }
                notified.await;
            }

            let request_id = state.next_request_id.fetch_add(1, Ordering::Relaxed);
            let client = http_client.clone();
            let endpoint = endpoint.clone();
            let response_sender = response_sender.clone();

            let (prompt, input_length, output_length) = ctx.inflate(data_index);

            let in_flight_state = Arc::new(InFlightState {
                input_length,
                output_tokens_so_far: AtomicU64::new(0),
                first_token_time_ms: AtomicU64::new(0),
            });
            state
                .in_flight
                .lock()
                .unwrap()
                .insert(request_id, Arc::clone(&in_flight_state));
            state.bs_active.fetch_add(1, Ordering::Relaxed);

            let task_state = Arc::clone(&state);
            let request_handle = spawn(async move {
                let json_body = A::request_json_body(prompt, input_length, output_length, stream);
                let s_time = get_timestamp();
                let result = post_with_timeout::<A>(
                    client,
                    endpoint.as_str(),
                    json_body.to_string(),
                    Duration::from_secs(timeout_secs_upon_slo(
                        output_length, ttft_slo, tpot_slo,
                    )),
                    stream,
                    Some(in_flight_state),
                )
                .await;

                // Accumulate output tokens into global counter
                let out_tokens = task_state
                    .in_flight
                    .lock()
                    .unwrap()
                    .get(&request_id)
                    .map(|s| s.output_tokens_so_far.load(Ordering::Relaxed))
                    .unwrap_or(0);
                task_state
                    .global_token_counter
                    .fetch_add(out_tokens, Ordering::Relaxed);

                // Remove from in-flight set, decrement counter, notify dispatch
                task_state.in_flight.lock().unwrap().remove(&request_id);
                task_state.bs_active.fetch_sub(1, Ordering::Relaxed);
                task_state.notify.notify_one();

                match result {
                    Ok(mut metrics) => {
                        let e_time = get_timestamp();
                        metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                        metrics.insert("s_time_drift".to_string(), "0.000".to_string());
                        metrics.insert("e_time".to_string(), format!("{e_time:.3}"));
                        metrics.insert("input_length".to_string(), input_length.to_string());
                        metrics.insert(
                            "output_length".to_string(),
                            output_length.to_string(),
                        );
                        let span_time = e_time - s_time;
                        metrics
                            .insert("span_time".to_string(), format!("{span_time:.3}"));
                        if output_length > 0
                            && metrics
                                .get("status")
                                .map(|s| s.starts_with('2'))
                                .unwrap_or(false)
                        {
                            let normalized_e2e = span_time / output_length as f64;
                            metrics.insert(
                                "normalized_e2e".to_string(),
                                format!("{normalized_e2e:.3}"),
                            );
                        }
                        response_sender.send(metrics).unwrap();
                    }
                    Err(RequestError::Timeout) => {
                        let e_time = get_timestamp();
                        let mut metrics = BTreeMap::<String, String>::from([(
                            "status".to_owned(),
                            "timeout".to_owned(),
                        )]);
                        metrics.insert("s_time".to_string(), format!("{s_time:.3}"));
                        metrics.insert("s_time_drift".to_string(), "0.000".to_string());
                        metrics.insert("e_time".to_string(), format!("{e_time:.3}"));
                        metrics.insert("input_length".to_string(), input_length.to_string());
                        metrics.insert(
                            "output_length".to_string(),
                            output_length.to_string(),
                        );
                        let span_time = e_time - s_time;
                        metrics
                            .insert("span_time".to_string(), format!("{span_time:.3}"));
                        response_sender.send(metrics).unwrap();
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RequestError::Other(error)) => {
                        tracing::error!(
                            "Request#{data_index}::({input_length}|{output_length}) error: {error}",
                        );
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RequestError::StreamErr(error)) => {
                        tracing::error!(
                            "Request#{data_index}::({input_length}|{output_length}) stream error: {error}",
                        );
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            tx.send_async(request_handle).await.unwrap();
        }

        // Drain: wait for all in-flight to complete
        loop {
            if state.bs_active.load(Ordering::Relaxed) == 0 {
                break;
            }
            state.notify.notified().await;
        }
        interrupt_flag.store(true, Ordering::SeqCst);
        tracing::info!(
            "Feedback requester: all {} entries consumed.",
            ctx.dataset.len()
        );
    });

    handle
}

/// The report loop writes the metrics to a file in JSONL format.
///
/// Report loop exits when the response receiver is closed.
pub async fn report_loop(
    mut output_jsonl_file: File,
    mut summary_json_file: File,
    response_receiver: flume::Receiver<BTreeMap<String, String>>,
) {
    let mut buf_writer = BufWriter::new(&mut output_jsonl_file);
    let mut summary = SummaryStats::new();
    while let Ok(metrics) = response_receiver.recv_async().await {
        summary.record(&metrics);
        let line = serde_json::to_string(&metrics).unwrap();
        buf_writer.write_all(line.as_bytes()).await.unwrap();
        buf_writer.write_all(b"\n").await.unwrap();
        buf_writer.flush().await.unwrap();
    }
    if let Some(metrics) = summary.finalize() {
        let line = serde_json::to_string_pretty(&metrics).unwrap();
        summary_json_file.write_all(line.as_bytes()).await.unwrap();
        summary_json_file.write_all(b"\n").await.unwrap();
        summary_json_file.flush().await.unwrap();
    }
}

struct SummaryStats {
    total_requests: u64,
    success_requests: u64,
    total_output_tokens: u64,
    min_s_time: Option<f64>,
    max_e_time: Option<f64>,
    ttft_values: Vec<f64>,
    tpot_values: Vec<f64>,
    tps_values: Vec<f64>,
    e2e_values: Vec<f64>,
}

impl SummaryStats {
    fn new() -> Self {
        Self {
            total_requests: 0,
            success_requests: 0,
            total_output_tokens: 0,
            min_s_time: None,
            max_e_time: None,
            ttft_values: Vec::new(),
            tpot_values: Vec::new(),
            tps_values: Vec::new(),
            e2e_values: Vec::new(),
        }
    }

    fn record(&mut self, metrics: &BTreeMap<String, String>) {
        self.total_requests += 1;

        if let Some(status) = metrics.get("status") {
            if status
                .parse::<u16>()
                .map(|code| (200..300).contains(&code))
                .unwrap_or(false)
            {
                self.success_requests += 1;
            }
        }

        if let Some(output_length) = metrics.get("output_length").and_then(|v| v.parse().ok()) {
            self.total_output_tokens = self.total_output_tokens.saturating_add(output_length);
        }

        if let Some(s_time) = metrics.get("s_time").and_then(|v| v.parse().ok()) {
            self.min_s_time = Some(self.min_s_time.map_or(s_time, |min| min.min(s_time)));
        }
        if let Some(e_time) = metrics.get("e_time").and_then(|v| v.parse().ok()) {
            self.max_e_time = Some(self.max_e_time.map_or(e_time, |max| max.max(e_time)));
        }

        if let Some(ttft) = metrics.get("first_token_time").and_then(|v| v.parse().ok()) {
            self.ttft_values.push(ttft);
        }
        if let Some(normalized_e2e) = metrics.get("normalized_e2e").and_then(|v| v.parse::<f64>().ok()) {
            // Non-streaming: E2E span_time / output_length (includes TTFT, not true TPOT)
            self.tpot_values.push(normalized_e2e);
            if normalized_e2e > 0.0 {
                self.tps_values.push(1000.0 / normalized_e2e);
            }
        } else if let (Some(total_time), Some(token_count)) = (
            metrics.get("total_time").and_then(|v| v.parse::<f64>().ok()),
            metrics
                .get("token_count")
                .and_then(|v| v.parse::<f64>().ok()),
        ) {
            // Streaming: TPOT = decode_time / (actual_token_count - 1)
            // total_time = last_token - first_token spans (N-1) inter-token gaps
            if token_count > 1.0 {
                let tpot = total_time / (token_count - 1.0);
                self.tpot_values.push(tpot);
                if tpot > 0.0 {
                    self.tps_values.push(1000.0 / tpot);
                }
            }
        }
        if let Some(e2e) = metrics.get("span_time").and_then(|v| v.parse().ok()) {
            self.e2e_values.push(e2e);
        }
    }

    fn finalize(&mut self) -> Option<BTreeMap<String, String>> {
        if self.total_requests == 0 {
            return None;
        }

        let percentiles = METRIC_PERCENTILES
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[90, 95, 99]);

        let mut summary = BTreeMap::new();
        summary.insert(
            "requests_total".to_string(),
            self.total_requests.to_string(),
        );
        summary.insert(
            "requests_success".to_string(),
            self.success_requests.to_string(),
        );
        summary.insert(
            "output_tokens_total".to_string(),
            self.total_output_tokens.to_string(),
        );

        let duration_ms = match (self.min_s_time, self.max_e_time) {
            (Some(start), Some(end)) if end >= start => end - start,
            _ => 0.0,
        };
        summary.insert("duration_ms".to_string(), format!("{duration_ms:.3}"));
        if duration_ms > 0.0 {
            let duration_secs = duration_ms / 1000.0;
            summary.insert(
                "throughput_rps".to_string(),
                format!("{:.3}", self.total_requests as f64 / duration_secs),
            );
            summary.insert(
                "throughput_tps".to_string(),
                format!("{:.3}", self.total_output_tokens as f64 / duration_secs),
            );
        }

        let ttft = compute_percentiles(&mut self.ttft_values, percentiles);
        for (percentile, value) in ttft {
            summary.insert(format!("ttft_p{percentile}_ms"), format_ms(value));
        }
        let tpot = compute_percentiles(&mut self.tpot_values, percentiles);
        for (percentile, value) in tpot {
            summary.insert(format!("tpot_p{percentile}_ms"), format_ms(value));
        }
        let e2e = compute_percentiles(&mut self.e2e_values, percentiles);
        for (percentile, value) in e2e {
            summary.insert(format!("e2e_p{percentile}_ms"), format_ms(value));
        }

        summary.insert(
            "ttft_mean_ms".to_string(),
            format_ms(mean(&self.ttft_values)),
        );
        summary.insert(
            "tpot_mean_ms".to_string(),
            format_ms(mean(&self.tpot_values)),
        );
        summary.insert(
            "tps_mean".to_string(),
            format!("{:.3}", mean(&self.tps_values)),
        );
        summary.insert(
            "e2e_mean_ms".to_string(),
            format_ms(mean(&self.e2e_values)),
        );

        Some(summary)
    }
}

fn compute_percentiles(values: &mut Vec<f64>, percentiles: &[u32]) -> Vec<(u32, f64)> {
    if values.is_empty() {
        return Vec::new();
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = values.len();
    percentiles
        .iter()
        .map(|percentile| {
            let idx = (len as f64 * (*percentile as f64 / 100.0)).ceil() as isize - 1;
            let idx = idx.max(0) as usize;
            let idx = idx.min(len - 1);
            (*percentile, values[idx])
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn format_ms(value: f64) -> String {
    format!("{:.3}", value)
}

#[cfg(test)]
#[cfg(feature = "prompt-text-hashed")]
mod tests {
    use super::*;
    use crate::{
        dataset::{BailianDataset, LLMTrace},
        token_sampler::TokenSampler,
    };
    use std::sync::Arc;
    use tokenizers::Tokenizer;
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
            Tokenizer::from_file("/Users/zdy/Workspace/Rust/request-sim/data/tokenizer.json")
                .unwrap(),
            "/Users/zdy/Workspace/Rust/request-sim/data/tokenizer_config.json".to_string(),
            4,   // num_producer
            128, // capacity
            16,  // block size
        ));

        // ====== 准备输出通道 ======
        let (tx, rx) = flume::unbounded();
        let output_file = File::create("tmp/inflate_latency.jsonl").await.unwrap();
        let summary_file = File::create("tmp/summary.json").await.unwrap();
        let reporter = tokio::spawn(report_loop(output_file, summary_file, rx));

        // ====== 测试循环 ======
        let iter = dataset.iter();
        for index in iter.take(10) {
            // 只测前10条
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
