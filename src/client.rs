use clap::Parser;
use request_sim::{
    dataset::{parse_dataset_type, Dataset, DatasetType},
    protocols::{DistserveProtocol, MockProtocol, StProtocol, VllmProtocol},
    requester::{
        create_gamma_interval_generator, report_loop, spawn_request_loop,
        spawn_request_loop_with_timestamp,
    },
};
use tokenizers::Tokenizer;
use tokio::{spawn, sync::oneshot};

#[derive(Parser)]
struct Args {
    /// Path to tokenizer file.
    #[clap(long, required = true)]
    tokenizer: String,

    /// Worker threads to use for tokio runime. Default is set to the number of cores.
    #[clap(long)]
    threads: Option<usize>,

    /// Endpoint URL to handle http request.
    /// For example, "http://localhost:8000/generate".
    #[clap(long, required = true)]
    endpoint: String,

    #[clap(long)]
    endpoints: Option<Vec<String>>,

    /// Protocol type. Either "st", "vllm", "distserve" or "mock".
    #[clap(long, short, required = true, value_parser = parse_protocol)]
    protocol: Protocol,

    /// Dataset type. Either "mooncake", "burstgpt", "mooncake_sampled", "azure", "uniform($input,$output)" or "mock".
    ///
    /// The uniform dataset requires input and output length arguments and its default request rate is 1.0 rps.
    ///
    /// To adjust the request rate, use the `request_rate` argument for non-replay mode and the `scale_factor` argument for replay mode instead.
    #[clap(long, short, required = true, value_parser = parse_dataset_type)]
    dataset_type: DatasetType,

    /// Path to dataset file. This argument is required only when dataset_type is not "mock" or "uniform".
    #[clap(long)]
    dataset_path: Option<String>,

    /// Path to second dataset. The second dataset file will be accessed only when dataset_type is "mooncake_sampled".
    #[clap(long)]
    second_dataset_path: Option<String>,

    /// If the replay_mode is enabled, the client will send requests following
    /// the sequence and input/output length of provided dataset above.
    ///
    /// Note that if the replay_mode is enabled, the cv will be ignored and the requests will not be shuffled.
    #[clap(long, default_value_t = false)]
    replay_mode: bool,

    /// Request rate (request per second). It only takes effect when `replay_mode` is disabled.
    #[clap(long)]
    request_rate: Option<f64>,

    /// Scale factor for the request rate. It only takes effect when `replay_mode` is enabled.
    ///
    /// For example, if the scale factor is 2 the client will send requests at twice the rate of the original data set.
    #[clap(long)]
    scale_factor: Option<f64>,

    /// Coefficient of variation of the request rate. It takes effect only when `replay_mode` is disabled.
    #[clap(long, default_value_t = 0.5)]
    cv: f64,

    /// Output path.
    #[clap(long, short, default_value = "./log/output.jsonl")]
    output_path: String,

    /// Requester run time.
    #[clap(long, short, default_value_t = 60)]
    time_in_secs: u64,

    /// If prefill_only is enabled, the output length will be set to the 1.
    #[clap(long, default_value_t = false)]
    prefill_only: bool,

    /// Truncate the request if the sum of input and output token length is greater than the specified value.
    #[clap(long)]
    truncate: Option<u64>,
}

fn parse_protocol(s: &str) -> Result<Protocol, String> {
    match s.to_lowercase().as_ref() {
        "st" => Ok(Protocol::St),
        "vllm" => Ok(Protocol::Vllm),
        "distserve" => Ok(Protocol::Distserve),
        "mock" => Ok(Protocol::Mock),
        _ => Err("Invalid protocol type".to_string()),
    }
}

#[derive(Debug, Clone, Copy)]
enum Protocol {
    St,
    Vllm,
    Distserve,
    Mock,
}

async fn async_main(args: Args) {
    let Args {
        tokenizer,
        threads: _,
        endpoint,
        endpoints,
        protocol,
        dataset_type,
        dataset_path,
        second_dataset_path,
        replay_mode,
        request_rate,
        scale_factor,
        cv,
        output_path,
        time_in_secs,
        prefill_only,
        truncate,
    } = args;

    let output_file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_path)
        .await
        .unwrap();

    let (response_tx, response_rx) = flume::unbounded();
    let dataset = match dataset_type {
        DatasetType::ProcessedCsv => {
            Dataset::load_processed_csv(&dataset_path.unwrap(), !replay_mode)
        }
        DatasetType::Mooncake => Dataset::load_mooncake_jsonl(&dataset_path.unwrap(), !replay_mode),
        DatasetType::Burstgpt => Dataset::load_burstgpt_csv(&dataset_path.unwrap(), !replay_mode),
        DatasetType::Azure => Dataset::load_azure_csv(&dataset_path.unwrap(), !replay_mode),
        DatasetType::MooncakeSampled => Dataset::load_mooncake_ts_burst_data(
            &dataset_path.unwrap(),
            &second_dataset_path.unwrap(),
            !replay_mode,
        ),
        DatasetType::Mock => Dataset::load_mock_dataset(),
        DatasetType::Uniform { input, output } => Dataset::load_uniform_dataset(input, output),
        DatasetType::CherryPickBurstgpt { start_ts, end_ts } => {
            Dataset::cherry_pick_burstgpt(&dataset_path.unwrap(), !replay_mode, start_ts, end_ts)
        }
    };
    let (stop_tx, stop_rx) = oneshot::channel();

    let protocol: Box<dyn request_sim::protocols::Protocol + Send> = match protocol {
        Protocol::St => Box::new(StProtocol::new(Tokenizer::from_file(tokenizer).unwrap())),
        Protocol::Vllm => Box::new(VllmProtocol::new(Tokenizer::from_file(tokenizer).unwrap())),
        Protocol::Distserve => Box::new(DistserveProtocol::new(
            Tokenizer::from_file(tokenizer).unwrap(),
        )),
        Protocol::Mock => Box::new(MockProtocol),
    };

    tracing::info!("Client start");
    let requester_handle = if replay_mode {
        spawn_request_loop_with_timestamp(
            endpoint,
            endpoints,
            dataset,
            prefill_only,
            truncate,
            protocol,
            scale_factor.unwrap(),
            response_tx,
            stop_rx,
        )
    } else {
        spawn_request_loop(
            endpoint,
            endpoints,
            dataset,
            prefill_only,
            truncate,
            protocol,
            create_gamma_interval_generator(request_rate.unwrap(), cv),
            response_tx,
            stop_rx,
        )
    };

    let reporter_handle = spawn(report_loop(output_file, response_rx));

    tokio::time::sleep(tokio::time::Duration::from_secs(time_in_secs)).await;
    stop_tx.send(()).unwrap();

    requester_handle.await.unwrap();
    reporter_handle.await.unwrap();
}

fn main() {
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    let args = Args::parse();
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    match args.threads {
        Some(threads) => builder.worker_threads(threads),
        None => &mut builder,
    }
    .enable_all()
    .build()
    .unwrap()
    .block_on(async_main(args));
}
