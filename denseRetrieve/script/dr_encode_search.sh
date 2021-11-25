#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=dr_encSr
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_dr_encSr.txt
#SBATCH -e log/erro_dr_encSr.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

model_list="biot5"
for model_name in $model_list
do
#model_name="biot5"
Base="denseRetrieve/data/ct21"
modelDir="denseRetrieve/models"
model_path=$modelDir"/ance/"$model_name
token_dir=$Base"/ance_token"
encodingPath=$Base"/ance_encoding/"$model_name
outRankFile=$Base"/ranking/"$model_name".txt"

mkdir -p $encodingPath
srun python -m tevatron.driver.encode \
  --output_dir $encodingPath \
  --model_name_or_path $model_path \
  --fp16 \
  --q_max_len 512 \
  --encode_is_qry \
  --per_device_eval_batch_size 1 \
  --encode_in_path $token_dir/query/query.json \
  --encoded_save_path $encodingPath/qry.pt

srun python -m tevatron.driver.encode \
  --output_dir $encodingPath \
  --model_name_or_path $model_path \
  --fp16 \
  --p_max_len 512 \
  --per_device_eval_batch_size 128 \
  --encode_in_path $token_dir/corpus/collection.json \
  --encoded_save_path $output_dir/collection.pt

mkdir -p $rankingPath

srun python -m tevatron.faiss_retriever \
  --query_reps $encodingPath/qry.pt \
  --passage_reps $encodingPath/collection.pt \
  --depth 2000 \
  --save_ranking_to $outRankFile

#srun python denseRetrieve/preprocess/util.py --file $outRankFile

done