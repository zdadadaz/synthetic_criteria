#!/bin/bash
Base="denseRetrieve/data/bioMSmarco"
TOKENIZER="roberta-base"
TOKENIZER_ID=$Base"/ance_token"
queryFile="../../data/bio-MSmarco/queries.train.tsv"
corpusFile="../../data/bio-MSmarco/collection.tsv"
python denseRetrieve/preprocess/tokenize_queries.py --truncate 512 --tokenizer_name $TOKENIZER --query_file $queryFile --save_to $TOKENIZER_ID/query/query.json
python denseRetrieve/preprocess/tokenize_passage.py --truncate 512 --tokenizer_name $TOKENIZER --file $corpusFile --save_to $TOKENIZER_ID/corpus
