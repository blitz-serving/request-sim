#!/bin/bash

proxy_off

tokenizer=/nvme/huggingface/hub/opt-1.3b/tokenizer.json
endpoint="http://127.0.0.1:5060/chat"
dataset_path=/nvme/wht/dataset/mooncake_trace.jsonl
data_type=mooncake
protocol=vllm

cargo run --bin client -- --tokenizer $tokenizer --endpoint $endpoint --dataset-path $dataset_path --dataset-type $data_type --protocol $protocol