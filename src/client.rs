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

    /// Worker threads to use for tokio runtime. Default is set to the number of cores.
    #[clap(long)]
    threads: Option<usize>,

    /// Endpoint URL to handle http request.
    #[clap(long, required = true)]
    endpoint: String,

    /// Protocol type. Either "st", "vllm", "distserve" or "mock".
    #[clap(long, short, required = true, value_parser = parse_protocol)]
    protocol: Protocol,

    /// Dataset type. Either "mooncake", "burstgpt" "mooncake_sampled" or "mock".
    #[clap(long, short, required = true, value_parser = parse_dataset_type)]
    dataset_type: DatasetType,

    /// Path to dataset file. The dataset file will be accessed only when dataset_type is not "mock".
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

    /// Error log path.
    #[clap(long, short, default_value = "./log/error.log")]
    error_log_path: String,

    /// Requester run time.
    #[clap(long, short, default_value_t = 60)]
    time_in_secs: u64,

    /// If prefill_only is enabled, the output length will be set to the 1.
    #[clap(long, default_value_t = false)]
    prefill_only: bool,

    /// Only used when dataset_type is "fake".
    #[clap(long)]
    fake_input_length: Option<u64>,

    /// Only used when dataset_type is "fake".
    #[clap(long)]
    fake_interval_ms: Option<u64>
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
        "azure" => Ok(DatasetType::Azure),
        "mooncake_sampled" => Ok(DatasetType::MooncakeSampled),
        "fake" => Ok(DatasetType::Fake),
        "mock" => Ok(DatasetType::Mock),
        _ => Err("Invalid dataset type.".to_string()),
    }
}

#[derive(Debug, Clone, Copy)]
enum DatasetType {
    Mooncake,
    Burstgpt,
    Azure,
    MooncakeSampled,
    Mock,
    Fake,
}

async fn async_main(args: Args) {
    let Args {
        tokenizer,
        threads: _,
        endpoint,
        protocol,
        dataset_type,
        dataset_path,
        second_dataset_path,
        replay_mode,
        request_rate,
        scale_factor,
        cv,
        output_path,
        error_log_path,
        time_in_secs,
        prefill_only,
        fake_input_length,
        fake_interval_ms,
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
        DatasetType::Mooncake => {
            Dataset::load_mooncake_jsonl(&dataset_path.unwrap(), !replay_mode, prefill_only)
        }
        DatasetType::Burstgpt => {
            Dataset::load_burstgpt_csv(&dataset_path.unwrap(), !replay_mode, prefill_only)
        }
        DatasetType::Azure => {
            Dataset::load_azure_csv(&dataset_path.unwrap(), !replay_mode, prefill_only)
        }
        DatasetType::MooncakeSampled => Dataset::load_mooncake_ts_burst_data(
            &dataset_path.unwrap(),
            &second_dataset_path.unwrap(),
            !replay_mode,
            prefill_only,
        ),
        DatasetType::Fake => Dataset::load_fake_dataset(fake_interval_ms.unwrap(), fake_input_length.unwrap(), prefill_only),
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
                    scale_factor.unwrap(),
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    st_protocol,
                    create_gamma_interval_generator(request_rate.unwrap(), cv),
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
                    scale_factor.unwrap(),
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    vllm_protocol,
                    create_gamma_interval_generator(request_rate.unwrap(), cv),
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
                    scale_factor.unwrap(),
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    distserve_protocol,
                    create_gamma_interval_generator(request_rate.unwrap(), cv),
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
                    scale_factor.unwrap(),
                    response_tx,
                    stop_rx,
                )
            } else {
                spawn_request_loop(
                    endpoint,
                    dataset,
                    MockProtocol,
                    create_gamma_interval_generator(request_rate.unwrap(), cv),
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
