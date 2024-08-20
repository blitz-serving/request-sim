use clap::Parser;
use request_sim::{
    dataset::Dataset,
    protocols::{DistserveProtocol, MockProtocol, TgiProtocol, VllmProtocol},
    requester::{
        create_gamma_interval_generator, report_loop, spawn_request_loop,
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

    /// Dataset type. Either "mooncake", "burstgpt" or "mock".
    #[clap(long, required = true, value_parser = parse_dataset_type)]
    dataset_type: DatasetType,

    /// Protocol type. Either "tgi", "vllm", "distserve" or "mock".
    #[clap(long, short, default_value = "tgi",  value_parser = parse_protocol)]
    protocol: Protocol,

    /// Path to dataset file.
    #[clap(long, required = true)]
    dataset_path: String,

    /// Request rate (request per second). It always takes effect whether `replay_mode` is enabled or not.
    #[clap(long, short, default_value_t = 1.0)]
    request_rate: f64,

    /// Coefficient of variation of the request rate. It takes effect only when `replay_mode` is enabled.
    #[clap(long, short, default_value_t = 0.5)]
    cv: f64,

    /// Output path
    #[clap(long, short, default_value = "/tmp/output.jsonl")]
    output_path: String,

    /// Requester run time.
    #[clap(long, short, default_value_t = 60)]
    time_in_secs: u64,

    /// If the replay_mode is enabled, the client will send requests following
    /// the sequence and input/output length of provided dataset above.
    #[clap(long, short, default_value_t = false)]
    replay_mode: bool,
}

fn parse_protocol(s: &str) -> Result<Protocol, String> {
    match s.to_lowercase().as_ref() {
        "tgi" => Ok(Protocol::Tgi),
        "vllm" => Ok(Protocol::Vllm),
        "distserve" => Ok(Protocol::Distserve),
        "mock" => Ok(Protocol::Mock),
        _ => Err("Invalid protocol type".to_string()),
    }
}

#[derive(Debug, Clone, Copy)]
enum Protocol {
    Tgi,
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
        .open(output_path)
        .await
        .unwrap();

    let (response_tx, response_rx) = flume::unbounded();
    let dataset = match dataset_type {
        DatasetType::Mooncake => Dataset::load_mooncake_jsonl(dataset_path.as_str(), true),
        DatasetType::Burstgpt => Dataset::load_burstgpt_csv(dataset_path.as_str(), true),
        DatasetType::Mock => Dataset::load_mock_dataset(),
    };
    let (stop_tx, stop_rx) = oneshot::channel();
    let requester_handle: JoinHandle<()> = match protocol {
        Protocol::Tgi => {
            let tgi_protocol = TgiProtocol::new(Tokenizer::from_file(tokenizer).unwrap());
            if replay_mode {
                spawn_request_loop_with_timestamp(
                    endpoint,
                    dataset,
                    tgi_protocol,
                    request_rate,
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    tgi_protocol,
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
