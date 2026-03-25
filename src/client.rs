use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use clap::Parser;
use request_sim::apis::{OpenAIApi, SglApi, AIBRIX_ROUTE_STRATEGY, METRIC_PERCENTILES};
use request_sim::cache::PromptCache;
use request_sim::{
    apis::{TGIApi, MODEL_NAME},
    dataset::LLMTrace,
    requester::{
        report_loop, spawn_request_loop_debug, spawn_request_loop_feedback,
        spawn_request_loop_random_process, spawn_request_loop_with_timestamp, ArrivalProcess,
        RequestContext,
    },
};
#[cfg(not(feature = "prompt-text-plain"))]
use request_sim::{
    dataset::{BailianDataset, MooncakeDataset},
    token_sampler::TokenSampler,
};
#[cfg(feature = "prompt-text-plain")]
use request_sim::dataset::MiniMaxDataset;
#[cfg(not(feature = "prompt-text-plain"))]
use tokenizers::Tokenizer;
use tokio::spawn;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::fmt::{self, format::FmtSpan};
use tracing_subscriber::{prelude::*, Layer, Registry};

#[derive(Parser)]
#[command(rename_all = "kebab-case")]
struct Args {
    /// Path to tokenizer file.
    #[cfg(not(feature = "prompt-text-plain"))]
    #[clap(long, required = true)]
    tokenizer: String,

    #[cfg(not(feature = "prompt-text-plain"))]
    #[clap(long, required = true)]
    tokenizer_config: String,

    /// Number of producer threads in TokenSampler.
    #[cfg(not(feature = "prompt-text-plain"))]
    #[clap(long)]
    num_producer: Option<usize>,

    /// Capacity of the channel between producers and consumers in TokenSampler.
    #[cfg(not(feature = "prompt-text-plain"))]
    #[clap(long)]
    channel_capacity: Option<usize>,

    /// Worker threads to use for tokio runime. Default is set to the number of cores.
    #[clap(long)]
    threads: Option<usize>,

    /// Endpoint URL to handle http request.
    /// For example, "http://localhost:8000/generate".
    #[clap(long, required = true)]
    endpoint: String,

    /// LLM API server type. Either "tgi" (text-generation-inference), or "distserve"
    #[clap(long, short, required = true)]
    api: String,

    /// Dataset type. Either "bailian", "mooncake", "azure", "uniform($input,$output)".
    ///
    /// The uniform dataset requires input and output length arguments and its default request rate is 1.0 rps.
    ///
    /// To adjust the request rate, use the `request_rate` argument for non-replay mode and the `scale_factor` argument for replay mode instead.
    #[clap(long, short, required = true)]
    dataset: String,

    /// Path to dataset file. This argument is required only when dataset_type is not "mock" or "uniform".
    #[clap(long, required = true)]
    dataset_path: Option<String>,

    /// Scale factor for the request rate. It only takes effect when `replay_mode` is enabled.
    ///
    /// For example, if the scale factor is 2 the client will send requests at twice the rate of the original data set.
    #[clap(long)]
    scale_factor: Option<f64>,

    /// Output path.
    #[clap(long, short, default_value = "./log/output.jsonl")]
    output_path: String,

    /// Summary output path (JSON).
    #[clap(long)]
    summary_path: Option<String>,

    /// Tracing path. Only used by tracing
    #[clap(long)]
    tracing_path: Option<String>,

    /// Requester run time.
    #[clap(long, short, default_value_t = 60)]
    time_in_secs: u64,

    /// Used for OpenAI API
    #[clap(long)]
    model_name: Option<String>,

    /// Used for AIBrix routing strategy
    #[clap(long)]
    aibrix_route: Option<String>,

    /// TTFT SLO in seconds
    #[clap(long, default_value_t = 5.0)]
    ttft_slo: f32,

    /// TPOT SLO in seconds
    #[clap(long, default_value_t = 0.06)]
    tpot_slo: f32,

    /// Enable streaming mode for API requests
    #[clap(long, default_value_t = false)]
    stream: bool,

    /// Percentiles (comma-separated) to report for latency metrics, e.g. "90,95,99"
    #[clap(long, value_delimiter = ',', default_value = "90,95,99")]
    metric_percentile: Vec<u32>,

    #[clap(long)]
    early_stop_error_threshold: Option<u32>,

    /// Cache mode for pre-generated prompts.
    ///   none  — no pre-generation, inflate inline via TokenSampler channels (default)
    ///   tmpfs — pre-generate all prompts to tmpfs (default: /dev/shm/request-sim-cache.bin)
    ///   file  — pre-generate all prompts to disk (default: ./request-sim-cache.bin)
    #[clap(long, default_value = "none")]
    cache: String,

    /// Path to the cache file. Defaults depend on --cache mode:
    ///   tmpfs → /dev/shm/request-sim-cache.bin
    ///   file  → ./request-sim-cache.bin
    #[clap(long)]
    cache_path: Option<String>,

    /// Begin timestamp (ms) of the trace time range to replay.
    /// Only requests with trace timestamp >= this value will be replayed.
    #[clap(long)]
    begin_time: Option<u64>,

    /// End timestamp (ms) of the trace time range to replay.
    /// Only requests with trace timestamp <= this value will be replayed.
    #[clap(long)]
    end_time: Option<u64>,

    /// Request dispatch mode: trace-replay, random-process, or feedback.
    #[clap(long, default_value = "trace-replay")]
    mode: String,

    /// Arrival process for random-process mode: poisson or uniform.
    #[clap(long)]
    arrival: Option<String>,

    /// Request rate (requests/second) for random-process mode.
    #[clap(long)]
    rate: Option<f64>,

    /// Target batch size (concurrent in-flight requests) for feedback mode.
    /// BS=1 is queueing-theory closed-loop (send one, await, send next).
    #[clap(long, default_value_t = 1)]
    target_bs: usize,
}

fn validate_config(args: &Args) {
    match args.mode.as_str() {
        "trace-replay" => {
            assert!(
                args.scale_factor.is_some(),
                "--scale-factor is required for trace-replay mode"
            );
        }
        "random-process" => {
            assert!(
                args.arrival.is_some(),
                "--arrival (poisson|uniform) is required for random-process mode"
            );
            assert!(
                args.rate.is_some(),
                "--rate is required for random-process mode"
            );
            let arrival = args.arrival.as_ref().unwrap();
            assert!(
                arrival == "poisson" || arrival == "uniform",
                "--arrival must be 'poisson' or 'uniform', got '{arrival}'"
            );
            let rate = args.rate.unwrap();
            assert!(rate > 0.0, "--rate must be positive, got {rate}");
        }
        "feedback" => {
            assert!(
                args.target_bs >= 1,
                "--target-bs must be >= 1, got {}",
                args.target_bs
            );
        }
        other => panic!(
            "Invalid --mode: '{other}'. Must be trace-replay, random-process, or feedback"
        ),
    }
    // release-with-debug always uses trace-replay
    if args.api.to_lowercase() == "release-with-debug" && args.mode != "trace-replay" {
        panic!("--api release-with-debug only supports --mode trace-replay");
    }
}

async fn async_main(args: Args) -> Result<(), i32> {
    validate_config(&args);

    let Args {
        #[cfg(not(feature = "prompt-text-plain"))]
        tokenizer,
        #[cfg(not(feature = "prompt-text-plain"))]
        tokenizer_config,
        #[cfg(not(feature = "prompt-text-plain"))]
        num_producer,
        #[cfg(not(feature = "prompt-text-plain"))]
        channel_capacity,
        threads: _,
        endpoint,
        api,
        dataset,
        dataset_path,
        scale_factor,
        output_path,
        summary_path,
        tracing_path: _,
        time_in_secs,
        model_name,
        aibrix_route,
        ttft_slo,
        tpot_slo,
        stream,
        metric_percentile,
        early_stop_error_threshold,
        cache,
        cache_path,
        begin_time,
        end_time,
        mode,
        arrival,
        rate,
        target_bs,
    } = args;

    let mut metric_percentile = metric_percentile;
    if metric_percentile.is_empty() {
        metric_percentile = vec![90, 95, 99];
    }
    metric_percentile.sort_unstable();
    metric_percentile.dedup();
    for percentile in &metric_percentile {
        assert!(
            (1..=100).contains(percentile),
            "Invalid metric percentile: {percentile}"
        );
    }
    METRIC_PERCENTILES.get_or_init(|| metric_percentile);

    let output_file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_path)
        .await
        .unwrap();
    let summary_path = summary_path.unwrap_or_else(|| format!("{output_path}.summary.json"));
    let summary_file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&summary_path)
        .await
        .unwrap();

    // --- Dataset dispatch ---
    #[cfg(not(feature = "prompt-text-plain"))]
    let block_size;

    let dataset: Pin<Box<dyn LLMTrace>> = match dataset.to_lowercase().as_str() {
        #[cfg(not(feature = "prompt-text-plain"))]
        "mooncake" => {
            let mut dataset: Pin<Box<MooncakeDataset>> = Box::pin(MooncakeDataset::new());
            block_size = 512;
            dataset.load(
                dataset_path
                    .expect("A dataset path must be provided in replay mode!")
                    .as_str(),
            );
            dataset
        }
        #[cfg(not(feature = "prompt-text-plain"))]
        "bailian" => {
            let mut dataset = Box::pin(BailianDataset::new());
            block_size = 16;
            dataset.load(
                dataset_path
                    .expect("A dataset path must be provided in replay mode!")
                    .as_str(),
            );
            dataset
        }
        #[cfg(feature = "prompt-text-plain")]
        "minimax" => {
            let mut dataset = Box::pin(MiniMaxDataset::new());
            dataset.load(
                dataset_path
                    .expect("A dataset path must be provided for minimax dataset!")
                    .as_str(),
            );
            dataset
        }
        _ => panic!("Invalid dataset type"),
    };

    let (tx, rx) = flume::unbounded();
    let interrupt_flag = Arc::new(AtomicBool::new(false));

    // Create token sampler (hashed mode only)
    #[cfg(not(feature = "prompt-text-plain"))]
    let token_sampler = Arc::new(TokenSampler::new(
        Tokenizer::from_file(tokenizer).unwrap(),
        tokenizer_config,
        num_producer.unwrap_or(1),
        channel_capacity.unwrap_or(128),
        block_size,
    ));

    // Resolve cache mode and path
    let cache_mode = cache.to_lowercase();
    let prompt_cache: Option<Arc<PromptCache>> = match cache_mode.as_str() {
        "none" => None,
        "tmpfs" => {
            let path = cache_path
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/dev/shm/request-sim-cache.bin"));
            #[cfg(not(feature = "prompt-text-plain"))]
            let cache = PromptCache::load_or_generate(
                dataset.as_ref().get_ref(),
                token_sampler.as_ref(),
                &path,
            );
            #[cfg(feature = "prompt-text-plain")]
            let cache = PromptCache::load_or_generate(
                dataset.as_ref().get_ref(),
                &path,
            );
            Some(Arc::new(cache))
        }
        "file" => {
            let path = cache_path
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./request-sim-cache.bin"));
            #[cfg(not(feature = "prompt-text-plain"))]
            let cache = PromptCache::load_or_generate(
                dataset.as_ref().get_ref(),
                token_sampler.as_ref(),
                &path,
            );
            #[cfg(feature = "prompt-text-plain")]
            let cache = PromptCache::load_or_generate(
                dataset.as_ref().get_ref(),
                &path,
            );
            Some(Arc::new(cache))
        }
        _ => panic!("Invalid cache mode: {cache_mode}. Use 'none', 'tmpfs', or 'file'."),
    };

    let dataset: Arc<Pin<Box<dyn LLMTrace>>> = Arc::new(dataset);

    // Build RequestContext for new mode functions
    let ctx = RequestContext {
        dataset: dataset.clone(),
        #[cfg(not(feature = "prompt-text-plain"))]
        token_sampler: token_sampler.clone(),
        prompt_cache: prompt_cache.clone(),
    };

    // Parse arrival process (for random-process mode)
    let arrival_process = arrival.as_deref().map(|a| match a {
        "poisson" => ArrivalProcess::Poisson,
        "uniform" => ArrivalProcess::Uniform,
        _ => unreachable!(), // validated in validate_config
    });

    // Set up API globals
    let api_lower = api.to_lowercase();
    if api_lower == "openai" || api_lower == "aibrix" || api_lower == "sgl" {
        MODEL_NAME.get_or_init(|| model_name.unwrap());
    }
    if api_lower == "aibrix" {
        AIBRIX_ROUTE_STRATEGY.get_or_init(|| {
            let valid_route_strategies = ["prefix-cache", "prefix-cache-preble", "throughput"];
            let route_strategy = aibrix_route.unwrap();
            assert!(
                valid_route_strategies.contains(&route_strategy.as_str()),
                "Unsupported AIBrix routing strategy: {}",
                route_strategy.as_str()
            );
            route_strategy
        });
    }

    tracing::info!("Client start");

    // Termination logging
    match mode.as_str() {
        "random-process" => {
            tracing::info!(
                "random-process mode: cyclic dispatch, will terminate after {}s timeout",
                time_in_secs
            );
        }
        "feedback" => {
            tracing::info!(
                "feedback mode: target_bs={}, will terminate when dataset ({} entries) exhausted or {}s timeout",
                target_bs,
                dataset.len(),
                time_in_secs
            );
        }
        _ => {} // trace-replay: existing logging in spawn function is sufficient
    }

    let time_range = (begin_time, end_time);

    // Dispatch: first resolve API type parameter, then match mode
    let requester_handle = match api_lower.as_str() {
        "release-with-debug" => spawn_request_loop_debug::<TGIApi>(
            endpoint,
            dataset,
            #[cfg(not(feature = "prompt-text-plain"))]
            token_sampler,
            scale_factor.unwrap(),
            tx,
            interrupt_flag.clone(),
            prompt_cache,
            time_range,
        ),
        "tgi" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<TGIApi>(
                endpoint,
                dataset,
                #[cfg(not(feature = "prompt-text-plain"))]
                token_sampler,
                scale_factor.unwrap(),
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                prompt_cache,
                time_range,
            ),
            "random-process" => spawn_request_loop_random_process::<TGIApi>(
                endpoint,
                ctx,
                arrival_process.unwrap(),
                rate.unwrap(),
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
            ),
            "feedback" => spawn_request_loop_feedback::<TGIApi>(
                endpoint,
                ctx,
                target_bs,
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
            ),
            _ => unreachable!(),
        },
        "openai" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<OpenAIApi>(
                endpoint,
                dataset,
                #[cfg(not(feature = "prompt-text-plain"))]
                token_sampler,
                scale_factor.unwrap(),
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                prompt_cache,
                time_range,
            ),
            "random-process" => spawn_request_loop_random_process::<OpenAIApi>(
                endpoint,
                ctx,
                arrival_process.unwrap(),
                rate.unwrap(),
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
            ),
            "feedback" => spawn_request_loop_feedback::<OpenAIApi>(
                endpoint,
                ctx,
                target_bs,
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
            ),
            _ => unreachable!(),
        },
        "aibrix" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<OpenAIApi>(
                endpoint,
                dataset,
                #[cfg(not(feature = "prompt-text-plain"))]
                token_sampler,
                scale_factor.unwrap(),
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                prompt_cache,
                time_range,
            ),
            "random-process" => spawn_request_loop_random_process::<OpenAIApi>(
                endpoint,
                ctx,
                arrival_process.unwrap(),
                rate.unwrap(),
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
            ),
            "feedback" => spawn_request_loop_feedback::<OpenAIApi>(
                endpoint,
                ctx,
                target_bs,
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
            ),
            _ => unreachable!(),
        },
        "sgl" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<SglApi>(
                endpoint,
                dataset,
                #[cfg(not(feature = "prompt-text-plain"))]
                token_sampler,
                scale_factor.unwrap(),
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                prompt_cache,
                time_range,
            ),
            "random-process" => spawn_request_loop_random_process::<SglApi>(
                endpoint,
                ctx,
                arrival_process.unwrap(),
                rate.unwrap(),
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
            ),
            "feedback" => spawn_request_loop_feedback::<SglApi>(
                endpoint,
                ctx,
                target_bs,
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
            ),
            _ => unreachable!(),
        },
        _ => unimplemented!("Unsupported protocol type"),
    };
    let reporter_handle = spawn(report_loop(output_file, summary_file, rx));

    // start test!
    tokio::select! {
        // case 1: time up
        _ = tokio::time::sleep(tokio::time::Duration::from_secs(time_in_secs)) => {
            tracing::info!("Test finished by timeout");
            interrupt_flag.store(true, Ordering::SeqCst);
        }

        // case 2: interrupt from inner
        _ = async {
            while !interrupt_flag.load(Ordering::Relaxed) {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        } => {
            tracing::error!("Test early stop by interrupt_flag");
        }
    }

    let ret_val = requester_handle.await.unwrap();
    reporter_handle.await.unwrap();
    return ret_val;
}

fn main() -> Result<(), i32> {
    let args = Args::parse();
    let console_layer = fmt::layer()
        .compact()
        .with_target(false)
        .with_filter(filter_fn(|meta| {
            !meta.name().contains("inflate")
        }))
        .with_filter(LevelFilter::INFO);
    if args.tracing_path.is_some() && args.api.to_lowercase().as_str() == "release-with-debug" {
        let file = std::fs::File::create(args.tracing_path.as_ref().unwrap()).unwrap();
        let file_layer = fmt::layer()
            .with_writer(file)
            .with_ansi(false)
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .with_thread_ids(true)
            .with_filter(filter_fn(|meta| {
                meta.target().starts_with("inflate") || meta.target().starts_with("spin_rwlck")
            }))
            .with_filter(LevelFilter::DEBUG);
        Registry::default()
            .with(console_layer)
            .with(file_layer)
            .init();
    } else {
        Registry::default().with(console_layer).init();
    }

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    match args.threads {
        Some(threads) => builder.worker_threads(threads),
        None => builder.worker_threads(55),
    }
    .enable_all()
    .build()
    .unwrap()
    .block_on(async_main(args))
}
