#!/bin/bash

model=/nvme/huggingface/hub/opt-1.3b
port=5060
dataset_path=/nvme/wht/dataset/mooncake_trace.jsonl

cargo run --bin client --tokenizer $model --endpoint http://localhost:$port --dataset-path $dataset_path --dataset-type mooncake