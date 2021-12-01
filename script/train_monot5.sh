#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH -o log/out_t5.txt
#SBATCH -e log/erro_t5.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla-smx2:1

# FT medt5
#srun python ./crossEncoder/finetunet5.py --triples_path ../../data/bio-MSmarco_ps/tripple.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/medMST5_ps_model

#### FT tc
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/tc_model


### FT tc + medt5
srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc_63.tsv \
                                              --learning_rate 1e-3 \
                                              --output_model_path ./crossEncoder/models/t5base/tc_medMST5_model \
                                              --base_model ./crossEncoder/models/t5base/medMST5_ps_model


############### train T5 large
### FT bio
#srun python ./crossEncoder/finetunet5.py --triples_path ../../data/bio-MSmarco_ps/tripple.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/lg/medMST5_model \
#                                              --base_model castorini/monot5-large-msmarco-10k
#
### FT tc + medt5
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/lg/tc_medMST5_model \
#                                              --base_model ./crossEncoder/models/lg/medMST5_model
#
##### FT tc
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/lg/tc_model \
#                                              --base_model castorini/monot5-large-msmarco-10k

## inference
## FT medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/lg/medMST5_model \
#                                        --outname medt5_lg

## FT tc + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/tc_medMST5_model \
#                                        --outname tc_medt5 \
#                                        --batchsize 128

## FT tc
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/lg/tc_model \
#                                        --outname tc_lg \
#                                        --batchsize 128


#srun python crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv  --save_every_n_steps 10000 --output_model_path test_monoT5_model