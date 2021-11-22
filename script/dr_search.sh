#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=drs
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_drsearch.txt
#SBATCH -e log/erro_drsearch.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

Base="denseRetrieve/data/bioMSmarco"
TOKENIZER="roberta-base"
TOKENIZER_ID=$Base"/ance_token"


encodingPath=$Base"/ance_encoding"
rankBase=$Base"/ranking"
rankingPath=$rankBase"/intermediate"
outRankFile=$rankBase/ance_rank.txt

#mkdir -p $rankingPath
#
#for i in $(seq -f "%02g" 0 9)
#do
#python -m tevatron.faiss_retriever \
#  --query_reps $encodingPath/qry.pt \
#  --passage_reps $encodingPath/split${i}.pt \
#  --depth 1000 \
#  --save_ranking_to $rankingPath/split${i}
#done

#python -m tevatron.faiss_retriever.reducer \
python denseRetrieve/tevatron/src/tevatron/faiss_retriever/reducer.py \
  --score_dir $rankingPath \
  --query $encodingPath/qry.pt \
  --save_ranking_to $outRankFile

#python -m pdb denseRetrieve/tevatron/src/tevatron/faiss_retriever/reducer.py --score_dir denseRetrieve/data/bioMSmarco/ranking/intermediate --query denseRetrieve/data/bioMSmarco/ance_encoding/qry.pt --save_ranking_to denseRetrieve/data/bioMSmarco/ranking/ance_rank.txt