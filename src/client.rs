use clap::Parser;
use request_sim::{
    dataset::Dataset,
    protocols::{DistserveProtocol, MockProtocol, StProtocol, VllmProtocol},
    requester::{
        create_gamma_interval_generator, init_error_log, report_loop, spawn_request_loop,
        spawn_request_loop_with_timestamp,
    },
};
use tokenizers::Tokenizer;
use tokio::{spawn, sync::oneshot, task::JoinHandle};

#[derive(Parser)]
struct Args {
    /// Path to tokenizer file.
    #[clap(long, required = true)]
    tokenizer: String,

    /// Worker threads to use for tokio runtime.
    #[clap(long, default_value_t = 10)]
    threads: usize,

    /// Endpoint URL to handle http request.
    #[clap(long, required = true)]
    endpoint: String,

    /// Protocol type. Either "st", "vllm", "distserve" or "mock".
    #[clap(long, short, required = true, value_parser = parse_protocol)]
    protocol: Protocol,

    /// Dataset type. Either "mooncake", "burstgpt" or "mock".
    #[clap(long, short, required = true, value_parser = parse_dataset_type)]
    dataset_type: DatasetType,

    /// Path to dataset file. The dataset file will be accessed only when dataset_type is not "mock".
    #[clap(long)]
    dataset_path: Option<String>,

    /// If the replay_mode is enabled, the client will send requests following
    /// the sequence and input/output length of provided dataset above.
    ///
    /// Note that if the replay_mode is enabled, the cv will be ignored and the requests will not be shuffled.
    #[clap(long, short, default_value_t = false)]
    replay_mode: bool,

    /// Request rate (request per second). It always takes effect whether `replay_mode` is enabled or not.
    #[clap(long, default_value_t = 1.0)]
    request_rate: f64,

    /// Coefficient of variation of the request rate. It takes effect only when `replay_mode` is disabled.
    #[clap(long, default_value_t = 0.5)]
    cv: f64,

    /// Output path.
    #[clap(long, short, default_value = "./log/output.jsonl")]
    output_path: String,

    /// Error log path.
    #[clap(long, short, default_value = "./log/error.log")]
    error_log_path: String,

    /// Requester run time.
    #[clap(long, short, default_value_t = 60)]
    time_in_secs: u64,
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

fn parse_dataset_type(s: &str) -> Result<DatasetType, String> {
    match s.to_lowercase().as_ref() {
        "mooncake" => Ok(DatasetType::Mooncake),
        "burstgpt" => Ok(DatasetType::Burstgpt),
        "mock" => Ok(DatasetType::Mock),
        _ => Err("Invalid dataset type.".to_string()),
    }
}

#[derive(Debug, Clone, Copy)]
enum DatasetType {
    Mooncake,
    Burstgpt,
    Mock,
}

async fn async_main(args: Args) {
    let Args {
        tokenizer,
        threads: _,
        endpoint,
        request_rate,
        cv,
        output_path,
        error_log_path,
        dataset_type,
        dataset_path,
        time_in_secs,
        protocol,
        replay_mode,
    } = args;

    let output_file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_path)
        .await
        .unwrap();
    tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&error_log_path)
        .await
        .unwrap();
    init_error_log(error_log_path).await;

    let (response_tx, response_rx) = flume::unbounded();
    let dataset = match dataset_type {
        DatasetType::Mooncake => Dataset::load_mooncake_jsonl(&dataset_path.unwrap(), !replay_mode),
        DatasetType::Burstgpt => Dataset::load_burstgpt_csv(&dataset_path.unwrap(), !replay_mode),
        DatasetType::Mock => Dataset::load_mock_dataset(),
    };
    let (stop_tx, stop_rx) = oneshot::channel();
    let requester_handle: JoinHandle<()> = match protocol {
        Protocol::St => {
            let st_protocol = StProtocol::new(Tokenizer::from_file(tokenizer).unwrap());
            if replay_mode {
                spawn_request_loop_with_timestamp(
                    endpoint,
                    dataset,
                    st_protocol,
                    request_rate,
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    st_protocol,
                    create_gamma_interval_generator(request_rate, cv),
                    response_tx,
                    stop_rx,
                )
            }
        }
        Protocol::Vllm => {
            let vllm_protocol = VllmProtocol::new(Tokenizer::from_file(tokenizer).unwrap());
            if replay_mode {
                spawn_request_loop_with_timestamp(
                    endpoint,
                    dataset,
                    vllm_protocol,
                    request_rate,
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    vllm_protocol,
                    create_gamma_interval_generator(request_rate, cv),
                    response_tx,
                    stop_rx,
                )
            }
        }
        Protocol::Distserve => {
            let distserve_protocol =
                DistserveProtocol::new(Tokenizer::from_file(tokenizer).unwrap());
            if replay_mode {
                spawn_request_loop_with_timestamp(
                    endpoint,
                    dataset,
                    distserve_protocol,
                    request_rate,
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    distserve_protocol,
                    create_gamma_interval_generator(request_rate, cv),
                    response_tx,
                    stop_rx,
                )
            }
        }
        Protocol::Mock => {
            if replay_mode {
                spawn_request_loop_with_timestamp(
                    endpoint,
                    dataset,
                    MockProtocol,
                    request_rate,
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    MockProtocol,
                    create_gamma_interval_generator(request_rate, cv),
                    response_tx,
                    stop_rx,
                )
            }
        }
    };
    let reporter_handle = spawn(report_loop(output_file, response_rx));

    tokio::time::sleep(tokio::time::Duration::from_secs(time_in_secs)).await;
    stop_tx.send(()).unwrap();

    requester_handle.await.unwrap();
    reporter_handle.await.unwrap();
}

fn main() {
    let args = Args::parse();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async_main(args));
}
