#!/bin/bash
Base="denseRetrieve/data/ct21"
TOKENIZER="roberta-base"
TOKENIZER_ID=$Base"/ance_token"
file="./data/splits/clean_data_cfg_splits_42_ct21"
queryFile="../../data/TRECCT2021/topics2021.tsv"
corpusFile="./data/splits/clean_data_cfg_splits_42_ct21"
python denseRetrieve/preprocess/tokenize_queries.py --truncate 512 --tokenizer_name $TOKENIZER --query_file $queryFile --save_to $TOKENIZER_ID/query/query.json
python denseRetrieve/preprocess/tokenize_collection.py --pickle_file $corpusFile --output_dir $TOKENIZER_ID/corpus
