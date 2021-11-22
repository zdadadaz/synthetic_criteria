#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5_3b
#SBATCH -n 1
#SBATCH --time=60:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_3b.txt
#SBATCH -e log/erro_3b.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4

# FT medt5
#srun python ./crossEncoder/finetunet5.py --triples_path ../../data/bio-MSmarco_ps/tripple.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/medMST5_ps_model

#### FT tc
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/tc_model \
#
#
## FT tc + medt5
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/tc_medMST5_model \
#                                              --base_model ./crossEncoder/models/t5base/medMST5_ps_model

############## train T5 3B
## FT tc
srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv  \
                                         --learning_rate 1e-3 \
                                         --output_model_path ./crossEncoder/models/3b/tc_medMST5_3B_model\
                                         --base_model castorini/monot5-3b-med-msmarco \
                                         --per_device_train_batch_size 1 \
                                         --gradient_accumulation_steps 128



#srun python finetunet5.py --triples_path triples/triples.train.small.tsv  --save_every_n_steps 10000 --output_model_path monoT5_model