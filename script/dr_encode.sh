#!/bin/bash

Base="denseRetrieve/data/bioMSmarco"
output_dir=$Base"/ance_encoding"
model_path="../ance-ct/tevatron/checkpt/Passage_ANCE_FirstP_Checkpoint/"
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


for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \
  --output_dir $output_dir \
  --model_name_or_path $model_path \
  --fp16 \
  --p_max_len 512 \
  --per_device_eval_batch_size 128 \
  --encode_in_path $token_dir/corpus/split${i}.json \
  --encoded_save_path $output_dir/split${i}.pt
done