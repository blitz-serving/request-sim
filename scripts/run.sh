#!/bin/bash

model=/huggingface/hub/opt-1.3b/tokenizer.json
port=8499
dataset_path=/huggingface/datasets/mooncake_trace.jsonl
dataset_type=mooncake
threads=1


cargo install --path .

RUST_BACKTRACE=1 client --tokenizer $model --endpoint localhost:$port --dataset-path $dataset_path --dataset-type $dataset_type --threads $threads 