use clap::Parser;
use request_sim::{
    dataset::Dataset,
    protocols::{distserve_protocol::DistserveProtocol, tgi_protocol::TgiProtocol},
    requester::{create_gamma_interval_generator, report_loop, spawn_request_loop},
};
use tokenizers::Tokenizer;
use tokio::{spawn, sync::oneshot};

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

    /// Dataset type. Either "mooncake" or "burstgpt".
    #[clap(long, required = true)]
    dataset_type: String,

    /// Path to dataset file.
    #[clap(long, required = true)]
    dataset_path: String,

    /// Request rate.
    #[clap(long, short, default_value_t = 1.0)]
    request_rate: f64,

    /// Coefficient of variation of the request rate
    #[clap(long, short, default_value_t = 0.5)]
    cv: f64,

    /// Output path
    #[clap(long, short, default_value = "output.csv")]
    output_path: String,

    /// Requester run time.
    #[clap(long, short, default_value_t = 60)]
    time_in_secs: u64,
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
    } = args;

    let output_file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)
        .await
        .unwrap();

    let (tx, rx) = flume::unbounded();
    //let protocol = TgiProtocol::new(Tokenizer::from_file(tokenizer).unwrap());
    let protocol = DistserveProtocol::new(Tokenizer::from_file(tokenizer).unwrap());
    let dataset = match dataset_type.to_lowercase().as_str() {
        "mooncake" => Dataset::load_mooncake_jsonl(dataset_path.as_str()),
        "burstgpt" => Dataset::load_burstgpt_csv(dataset_path.as_str()),
        _ => panic!("Invalid dataset type"),
    };

    let interval_generator = create_gamma_interval_generator(request_rate, cv);
    let (stop_tx, stop_rx) = oneshot::channel();
    let handle_1 = spawn_request_loop(endpoint, dataset, protocol, interval_generator, tx, stop_rx);
    let handle_2 = spawn(report_loop(output_file, rx));

    tokio::time::sleep(tokio::time::Duration::from_secs(time_in_secs)).await;
    stop_tx.send(()).unwrap();

    handle_1.await.unwrap();
    handle_2.await.unwrap();
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
