#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=drtk
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_dr_tokenize.txt
#SBATCH -e log/erro_dr_tokenize.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

Base="denseRetrieve/data/ct21"
TOKENIZER="roberta-base"
TOKENIZER_ID=$Base"/ance_token"
file="./data/splits/clean_data_cfg_splits_63_ct21"
queryFile="../../data/TRECCT2021/topics2021.tsv"
corpusFile="./data/splits/clean_data_cfg_splits_63_ct21"
#python denseRetrieve/preprocess/tokenize_queries.py --truncate 512 --tokenizer_name $TOKENIZER --query_file $queryFile --save_to $TOKENIZER_ID/query/query.json
#python denseRetrieve/preprocess/tokenize_collection.py --pickle_file $corpusFile --output_dir $TOKENIZER_ID/corpus
