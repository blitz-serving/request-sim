use std::{pin::Pin, sync::Arc};

use clap::Parser;
use request_sim::{
    apis::TGIApi,
    dataset::{AzureDataset, BailianDataset, LLMTrace, MooncakeDataset},
    requester::{
        create_gamma_interval_generator, report_loop, spawn_request_loop,
        spawn_request_loop_with_timestamp,
    },
    token_sampler::TokenSampler,
};
use tokenizers::Tokenizer;
use tokio::{spawn, sync::broadcast};

#[derive(Parser)]
struct Args {
    /// Path to tokenizer file.
    #[clap(long, required = true)]
    tokenizer: String,

    #[clap(long, required = true)]
    tokenizer_config: String,

    /// Number of producer threads in TokenSampler.
    #[clap(long)]
    num_producer: Option<usize>,

    /// Capacity of the channel between producers and consumers in TokenSampler.
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
    #[clap(long)]
    dataset_path: Option<String>,

    /// Path to second dataset. The second dataset file will be accessed only when dataset_type is "mooncake_sampled".
    #[clap(long)]
    second_dataset_path: Option<String>,

    /// If the `replay_mode` is enabled, the client will send requests following
    /// the sequence and input/output length of provided dataset above.
    ///
    /// Note that if the `replay_mode` is enabled, the cv will be ignored and the requests will not be shuffled.
    #[clap(long, default_value_t = false)]
    replay_mode: bool,

    #[clap(long)]
    scale_replay_path: Option<String>,

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
}

async fn async_main(args: Args) -> Result<(), i32> {
    let Args {
        tokenizer,
        tokenizer_config,
        num_producer,
        channel_capacity,
        threads: _,
        endpoint,
        api,
        dataset,
        dataset_path,
        second_dataset_path,
        replay_mode,
        scale_replay_path,
        request_rate,
        scale_factor,
        cv,
        output_path,
        time_in_secs,
    } = args;

    let output_file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_path)
        .await
        .unwrap();

    let mut block_size = 0;

    let dataset: Pin<Box<dyn LLMTrace>> = match dataset.to_lowercase().as_str() {
        "mooncake" => {
            let mut dataset = Box::pin(MooncakeDataset::new());
            block_size = 128;
            dataset.load(
                dataset_path
                    .expect("A dataset path must be provided in replay mode!")
                    .as_str(),
            );
            dataset
        }
        "burstgpt" => {
            let mut dataset = Box::pin(AzureDataset::new());
            block_size = 16;
            dataset.load(
                dataset_path
                    .expect("A dataset path must be provided in replay mode!")
                    .as_str(),
            );
            dataset
        }
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
        "uniform" => {
            unimplemented!("Uniform length dataset is unimplemented!");
        }
        "azure" => {
            let mut dataset = Box::pin(AzureDataset::new());
            block_size = 16;
            dataset.load(
                dataset_path
                    .expect("A dataset path must be provided in replay mode!")
                    .as_str(),
            );
            dataset
        }
        _ => panic!("Invalid dataset type"),
    };

    let (tx, rx) = flume::unbounded();
    let (broadcast_tx, _rx) = broadcast::channel(1);

    tracing::info!("Client start");
    // TODO: check `spawn_request_loop_with_timestamp` API
    let requester_handle = match api.to_lowercase().as_str() {
        "tgi" => {
            let dataset: Arc<Pin<Box<dyn LLMTrace>>> = Arc::new(dataset);
            let token_sampler = Arc::new(TokenSampler::new(
                Tokenizer::from_file(tokenizer).unwrap(),
                tokenizer_config,
                num_producer.unwrap_or(1),
                channel_capacity.unwrap_or(128),
                block_size,
            ));
            spawn_request_loop_with_timestamp::<TGIApi>(
                endpoint,
                dataset,
                token_sampler,
                scale_factor.unwrap(),
                tx,
                broadcast_tx.clone(),
            )
        }
        _ => unimplemented!("Unsupported protocol type"),
    };
    let reporter_handle = spawn(report_loop(output_file, rx));

    // start test!
    tokio::time::sleep(tokio::time::Duration::from_secs(time_in_secs)).await;
    broadcast_tx.send(()).unwrap(); // terminate test

    let returnval = requester_handle.await.unwrap();
    reporter_handle.await.unwrap();
    return returnval;
}

fn main() -> Result<(), i32> {
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    let args = Args::parse();
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
