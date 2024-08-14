#!/bin/bash

#model=/huggingface/hub/opt-1.3b/tokenizer.json
#model=/huggingface/hub/opt-13b/tokenizer.json
model=/huggingface/hub/Llama-2-7b-hf/tokenizer.json
port=8499
#dataset_path=/huggingface/datasets/mooncake_trace.jsonl
#dataset_type=mooncake
dataset_path=/huggingface/datasets/burstgpt.csv
dataset_type=burstgpt
threads=56
req_rate=200
cv=2

CARGO_BUILD_JOBS=88 cargo install --path .

RUST_BACKTRACE=1 client --tokenizer $model --threads $threads --endpoint http://localhost:$port/generate --dataset-path $dataset_path --dataset-type $dataset_type --protocol distserve -t 900 --request-rate $req_rate --cv $cv