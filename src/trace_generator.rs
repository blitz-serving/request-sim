use std::io::Write;

use clap::Parser;
use request_sim::{dataset, requester::create_gamma_interval_generator};

fn main() {
    let args: Args = Args::parse();
    let dataset = match args.dataset_name.to_lowercase().as_str() {
        "mooncake" => dataset::Dataset::load_mooncake_jsonl(&args.dataset_path, true),
        "burstgpt" => dataset::Dataset::load_burstgpt_csv(&args.dataset_path, true),
        "mooncake_sampled" => dataset::Dataset::load_mooncake_ts_burst_data(&args.dataset_path, "/nvme/huggingface/datasets/burstgpt-v2.csv", true),
        _ => panic!("Invalid dataset name"),
    };
    let inverval_generator = create_gamma_interval_generator(args.request_rate, args.cv);
    let mut current_timestamp = 0.0;
    let file = std::fs::File::create(args.output_path).unwrap();
    let mut writer = std::io::BufWriter::new(file);

    writer
        .write("timestamp,input_token_length,output_token_length\n".as_bytes())
        .unwrap();
    for _ in 0..args.output_lines {
        let (input_token_length, output_token_length) = dataset.next_request();
        let record = format!(
            "{},{},{}\n",
            current_timestamp as u64, input_token_length, output_token_length
        );
        writer.write(record.as_bytes()).unwrap();
        current_timestamp += inverval_generator.interval_in_millis();
    }

    writer.flush().unwrap();
}

#[derive(Parser, Debug)]
pub struct Args {
    /// Average request rate
    #[clap(short, long)]
    request_rate: f64,

    /// CV of gamma distribution
    #[clap(long)]
    cv: f64,

    /// `mooncake` or `burstgpt`
    #[clap(long)]
    dataset_name: String,

    /// Path to the dataset
    #[clap(long)]
    dataset_path: String,

    /// Output path
    #[clap(long)]
    output_path: String,

    /// Number of output lines
    #[clap(long)]
    output_lines: usize,
}
