#!/bin/bash

Base="denseRetrieve/data/ct21"
output_dir=$Base"/ance_encoding/biot5"
model_path="denseRetrieve/models/ance/biot5/"
token_dir=$Base"/ance_token"

mkdir $output_dir
python -m tevatron.driver.encode \
  --output_dir $output_dir \
  --model_name_or_path $model_path \
  --fp16 \
  --q_max_len 512 \
  --encode_is_qry \
  --per_device_eval_batch_size 1 \
  --encode_in_path $token_dir/query/query.json \
  --encoded_save_path $output_dir/qry.pt


python -m tevatron.driver.encode \
  --output_dir $output_dir \
  --model_name_or_path $model_path \
  --fp16 \
  --p_max_len 512 \
  --per_device_eval_batch_size 128 \
  --encode_in_path $token_dir/corpus/collection.json \
  --encoded_save_path $output_dir/collection.pt
