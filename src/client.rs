use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use clap::Parser;
use request_sim::apis::{OaiApi, SglApi, AIBRIX_ROUTE_STRATEGY, AMADEUS_ID, CONTEXT_LENGTH, MAX_TOKENS_CAP, METRIC_PERCENTILES, RID_SOURCE, RidSource};
use request_sim::cache::PromptCache;
use request_sim::{
    apis::{TgiApi, MODEL_NAME},
    dataset::LLMTrace,
    requester::{
        report_loop, spawn_request_loop_debug, spawn_request_loop_feedback,
        spawn_request_loop_random_process, spawn_request_loop_with_timestamp, ArrivalProcess,
        ControllerConfig, RequestContext,
    },
};
use request_sim::apis::AmadeusApi;
#[cfg(feature = "prompt-text-hashed")]
use request_sim::{
    dataset::{BailianDataset, MooncakeDataset},
    token_sampler::TokenSampler,
};
#[cfg(feature = "prompt-text-plain")]
use request_sim::dataset::{OpenaiDataset, PlainTextDataset, AmadeusDataset};
#[cfg(feature = "prompt-text-hashed")]
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
    #[cfg(feature = "prompt-text-hashed")]
    #[clap(long, required = true)]
    tokenizer: String,

    #[cfg(feature = "prompt-text-hashed")]
    #[clap(long, required = true)]
    tokenizer_config: String,

    /// Number of producer threads in TokenSampler.
    #[cfg(feature = "prompt-text-hashed")]
    #[clap(long)]
    num_producer: Option<usize>,

    /// Capacity of the channel between producers and consumers in TokenSampler.
    #[cfg(feature = "prompt-text-hashed")]
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

    /// Maximum concurrent in-flight requests for feedback mode.
    /// Without constraints, acts as a static semaphore. With constraints, the AIMD
    /// controller probes upward toward this ceiling.
    #[clap(long, default_value_t = 1)]
    bs_limit: usize,

    /// Controller tick interval in seconds (feedback mode with constraints).
    #[clap(long, default_value_t = 0.2)]
    controller_interval: f64,

    /// Number of ticks to skip after each actuation before the next adjustment.
    #[clap(long, default_value_t = 1)]
    cooldown_ticks: u32,

    /// TPOT upper bound in ms. Activates AIMD controller when set.
    #[clap(long)]
    tpot_limit: Option<f64>,

    /// TPS lower bound (tokens/sec). Activates AIMD controller when set.
    #[clap(long)]
    tps_limit: Option<f64>,

    /// Total context tokens upper bound. Activates AIMD controller when set.
    #[clap(long)]
    all_tokens_limit: Option<u64>,

    /// Safety cap on output tokens. Applied as max_tokens when the dataset does
    /// not specify an output length (e.g. plaintext dataset). Ignored when the
    /// trace provides an explicit output_length.
    #[clap(long)]
    max_tokens: Option<u64>,

    /// Model context window size (in tokens). When set and a request's
    /// input_length + output_length exceeds this limit, min_tokens is omitted
    /// from the request body to avoid server-side errors. max_tokens is kept
    /// so the server can cap or reject as appropriate.
    #[clap(long)]
    context_length: Option<u64>,

    /// Source for request rid (content identity).
    ///   none         — no rid in request body (default)
    ///   content-hash — SHA256 of serialized messages, truncated to 16 hex chars
    #[clap(long, default_value = "none")]
    rid_source: String,

    /// Amadeus ID for the meta.amadeus_id field in amadeus API requests.
    #[clap(long)]
    amadeus_id: Option<i64>,

    /// Track engine output text and inject into subsequent turns' prompts.
    /// Fixes multi-turn KV cache mismatch in hashed mode (bailian dataset).
    /// Only applies to trace-replay mode. Incompatible with --cache.
    #[cfg(feature = "prompt-text-hashed")]
    #[clap(long, default_value_t = false)]
    track_output: bool,

    /// Controls inter-turn timing when --track-output is enabled.
    /// - preserve-gap: session start scaled, intra-session gaps preserved (default)
    /// - scale-gap: all times scaled proportionally, causality preserved
    /// - scale-all: all times scaled, no causality — most aggressive
    #[cfg(feature = "prompt-text-hashed")]
    #[clap(long, default_value = "preserve-gap")]
    track_output_timing: String,
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
                args.bs_limit >= 1,
                "--bs-limit must be >= 1, got {}",
                args.bs_limit
            );
            assert!(
                args.controller_interval > 0.0,
                "--controller-interval must be positive"
            );
            if args.tpot_limit.is_some() || args.tps_limit.is_some() {
                assert!(
                    args.stream,
                    "--tpot-limit and --tps-limit require --stream (real-time token observation)"
                );
            }
        }
        other => panic!(
            "Invalid --mode: '{other}'. Must be trace-replay, random-process, or feedback"
        ),
    }
    // release-with-debug always uses trace-replay
    if args.api.to_lowercase() == "release-with-debug" && args.mode != "trace-replay" {
        panic!("--api release-with-debug only supports --mode trace-replay");
    }

    #[cfg(feature = "prompt-text-hashed")]
    if args.track_output {
        assert!(
            args.mode == "trace-replay" || args.mode == "feedback",
            "--track-output only supports --mode trace-replay and --mode feedback. \
             random-process mode does not maintain conversation state."
        );
        if args.cache != "none" {
            tracing::warn!(
                "--track-output is incompatible with --cache (cache pre-generates all prompts \
                 before requests are sent, so runtime output injection cannot work). \
                 Cache will be disabled."
            );
        }
    }
}

async fn async_main(args: Args) -> Result<(), i32> {
    validate_config(&args);

    let Args {
        #[cfg(feature = "prompt-text-hashed")]
        tokenizer,
        #[cfg(feature = "prompt-text-hashed")]
        tokenizer_config,
        #[cfg(feature = "prompt-text-hashed")]
        num_producer,
        #[cfg(feature = "prompt-text-hashed")]
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
        bs_limit,
        controller_interval,
        cooldown_ticks,
        tpot_limit,
        tps_limit,
        all_tokens_limit,
        max_tokens,
        context_length,
        rid_source,
        amadeus_id,
        #[cfg(feature = "prompt-text-hashed")]
        track_output,
        #[cfg(feature = "prompt-text-hashed")]
        track_output_timing,
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
    MAX_TOKENS_CAP.get_or_init(|| max_tokens);
    CONTEXT_LENGTH.get_or_init(|| context_length);
    RID_SOURCE.get_or_init(|| match rid_source.as_str() {
        "none" => RidSource::None,
        "content-hash" => RidSource::ContentHash,
        other => panic!("Invalid --rid-source: '{other}'. Must be 'none' or 'content-hash'"),
    });

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
    #[cfg(feature = "prompt-text-hashed")]
    let block_size;

    let dataset: Pin<Box<dyn LLMTrace>> = match dataset.to_lowercase().as_str() {
        #[cfg(feature = "prompt-text-hashed")]
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
        #[cfg(feature = "prompt-text-hashed")]
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
        "plaintext" => {
            let mut dataset = Box::pin(PlainTextDataset::new());
            dataset.load(
                dataset_path
                    .expect("--dataset-path is required for plaintext dataset")
                    .as_str(),
            );
            dataset
        }
        #[cfg(feature = "prompt-text-plain")]
        "openai" => {
            let mut dataset = Box::pin(OpenaiDataset::new());
            dataset.load(
                dataset_path
                    .expect("A dataset path must be provided for openai dataset!")
                    .as_str(),
            );
            dataset
        }
        #[cfg(feature = "prompt-text-plain")]
        "amadeus-replay" => {
            let mut dataset = Box::pin(AmadeusDataset::new());
            dataset.load(
                dataset_path
                    .expect("--dataset-path is required for amadeus-replay dataset")
                    .as_str(),
            );
            dataset
        }
        _ => panic!("Invalid dataset type"),
    };

    let (tx, rx) = flume::unbounded();
    let interrupt_flag = Arc::new(AtomicBool::new(false));

    // Create token sampler (hashed mode only)
    #[cfg(feature = "prompt-text-hashed")]
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
            #[cfg(feature = "prompt-text-hashed")]
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
            #[cfg(feature = "prompt-text-hashed")]
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

    // Build output tracking state (hashed mode, bailian only)
    #[cfg(feature = "prompt-text-hashed")]
    let track_output_state: Option<request_sim::requester::TrackOutputState> = if track_output {
        let graph = dataset
            .build_conversation_graph()
            .expect(
                "--track-output requires a dataset with multi-turn conversation support \
                 (bailian). Other datasets (mooncake, plaintext, openai) do not have \
                 chat_id/parent_chat_id fields needed for conversation graph construction."
            );
        let tokenizer = token_sampler.get_tokenizer();
        let template = request_sim::dataset::TemplateRegistry::for_tokenizer_class("Qwen2Tokenizer");
        let state = request_sim::requester::TrackOutputState::new(
            graph,
            tokenizer,
            template,
            dataset.len(),
        );
        // Disable cache when tracking output
        Some(state)
    } else {
        None
    };

    #[cfg(feature = "prompt-text-hashed")]
    let timing_mode = request_sim::requester::TrackOutputTiming::from_str(&track_output_timing);

    // Build RequestContext for new mode functions
    let ctx = RequestContext {
        dataset: dataset.clone(),
        #[cfg(feature = "prompt-text-hashed")]
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
    if api_lower == "amadeus" {
        AMADEUS_ID.get_or_init(|| amadeus_id);
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
                "feedback mode: bs_limit={}, will terminate when dataset ({} entries) exhausted or {}s timeout",
                bs_limit,
                dataset.len(),
                time_in_secs
            );
        }
        _ => {} // trace-replay: existing logging in spawn function is sufficient
    }

    let time_range = (begin_time, end_time);

    // Dispatch: first resolve API type parameter, then match mode
    let requester_handle = match api_lower.as_str() {
        "release-with-debug" => spawn_request_loop_debug::<TgiApi>(
            endpoint,
            dataset,
            #[cfg(feature = "prompt-text-hashed")]
            token_sampler,
            scale_factor.unwrap(),
            tx,
            interrupt_flag.clone(),
            prompt_cache,
            time_range,
        ),
        "tgi" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<TgiApi>(
                endpoint,
                dataset,
                #[cfg(feature = "prompt-text-hashed")]
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
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
                #[cfg(feature = "prompt-text-hashed")]
                timing_mode,
            ),
            "random-process" => spawn_request_loop_random_process::<TgiApi>(
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
            "feedback" => spawn_request_loop_feedback::<TgiApi>(
                endpoint,
                ctx,
                ControllerConfig {
                    bs_limit,
                    interval_secs: controller_interval,
                    cooldown_ticks,
                    tpot_limit_ms: tpot_limit,
                    tps_limit,
                    all_tokens_limit,
                },
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
            ),
            _ => unreachable!(),
        },
        "openai" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<OaiApi>(
                endpoint,
                dataset,
                #[cfg(feature = "prompt-text-hashed")]
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
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
                #[cfg(feature = "prompt-text-hashed")]
                timing_mode,
            ),
            "random-process" => spawn_request_loop_random_process::<OaiApi>(
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
            "feedback" => spawn_request_loop_feedback::<OaiApi>(
                endpoint,
                ctx,
                ControllerConfig {
                    bs_limit,
                    interval_secs: controller_interval,
                    cooldown_ticks,
                    tpot_limit_ms: tpot_limit,
                    tps_limit,
                    all_tokens_limit,
                },
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
            ),
            _ => unreachable!(),
        },
        "aibrix" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<OaiApi>(
                endpoint,
                dataset,
                #[cfg(feature = "prompt-text-hashed")]
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
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
                #[cfg(feature = "prompt-text-hashed")]
                timing_mode,
            ),
            "random-process" => spawn_request_loop_random_process::<OaiApi>(
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
            "feedback" => spawn_request_loop_feedback::<OaiApi>(
                endpoint,
                ctx,
                ControllerConfig {
                    bs_limit,
                    interval_secs: controller_interval,
                    cooldown_ticks,
                    tpot_limit_ms: tpot_limit,
                    tps_limit,
                    all_tokens_limit,
                },
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
            ),
            _ => unreachable!(),
        },
        "sgl" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<SglApi>(
                endpoint,
                dataset,
                #[cfg(feature = "prompt-text-hashed")]
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
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
                #[cfg(feature = "prompt-text-hashed")]
                timing_mode,
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
                ControllerConfig {
                    bs_limit,
                    interval_secs: controller_interval,
                    cooldown_ticks,
                    tpot_limit_ms: tpot_limit,
                    tps_limit,
                    all_tokens_limit,
                },
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
            ),
            _ => unreachable!(),
        },
        "amadeus" => match mode.as_str() {
            "trace-replay" => spawn_request_loop_with_timestamp::<AmadeusApi>(
                endpoint,
                dataset,
                #[cfg(feature = "prompt-text-hashed")]
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
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
                #[cfg(feature = "prompt-text-hashed")]
                timing_mode,
            ),
            "random-process" => spawn_request_loop_random_process::<AmadeusApi>(
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
            "feedback" => spawn_request_loop_feedback::<AmadeusApi>(
                endpoint,
                ctx,
                ControllerConfig {
                    bs_limit,
                    interval_secs: controller_interval,
                    cooldown_ticks,
                    tpot_limit_ms: tpot_limit,
                    tps_limit,
                    all_tokens_limit,
                },
                tx,
                interrupt_flag.clone(),
                ttft_slo,
                tpot_slo,
                stream,
                early_stop_error_threshold,
                #[cfg(feature = "prompt-text-hashed")]
                track_output_state.clone(),
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
