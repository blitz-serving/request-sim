#!/bin/bash

model=/huggingface/hub/opt-1.3b/tokenizer.json
port=8499
dataset_path=/huggingface/datasets/mooncake_trace.jsonl
dataset_type=mooncake
threads=1


CARGO_BUILD_JOBS=88 cargo install --path .

RUST_BACKTRACE=1 client --tokenizer $model --endpoint http://localhost:$port/generate --dataset-path $dataset_path --dataset-type $dataset_type --threads $threads --protocol distserve -t 10