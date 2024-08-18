#!/bin/bash

#model=/huggingface/hub/opt-1.3b/tokenizer.json
#model=/huggingface/hub/opt-13b/tokenizer.json
model=/huggingface/hub/Llama-2-7b-hf/tokenizer.json
port=8499
#dataset_path=/huggingface/datasets/mooncake_trace.jsonl
#dataset_type=mooncake
dataset_path=/huggingface/datasets/burstgpt-v2.csv
dataset_type=burstgpt
threads=56
req_rate=11
cv=1.5

CARGO_BUILD_JOBS=88 cargo install --path .

RUST_BACKTRACE=1 client --tokenizer $model --endpoint http://localhost:$port/generate --dataset-path $dataset_path --dataset-type $dataset_type --protocol distserve -t 1200 --request-rate $req_rate --cv $cv