#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5_3b2
#SBATCH -n 1
#SBATCH --time=60:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH -o log/out_3b2.txt
#SBATCH -e log/erro_3b2.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

## FT tc + medt5
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/tc_medMST5_model \
#                                              --base_model ./crossEncoder/models/t5base/medMST5_ps_model

############## train T5 3B
# FT medt5
#srun python ./crossEncoder/finetunet5.py --triples_path ../../data/bio-MSmarco_ps/tripple.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/3b/medMST5_model \
#                                              --base_model castorini/monot5-3b-msmarco \
#                                              --per_device_train_batch_size 1 \
#                                              --gradient_accumulation_steps 42
### FT tc
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv  \
#                                         --learning_rate 1e-3 \
#                                         --output_model_path ./crossEncoder/models/3b/tc_med_model\
#                                         --per_device_train_batch_size 8 \
#                                         --gradient_accumulation_steps 16 \
#                                         --base_model castorini/monot5-3b-med-msmarco
##                                         --gradient_checkpointing True
##                                         --base_model castorini/monot5-large-msmarco-10k \

srun python ./crossEncoder/finetunet5_tc_val.py --triples_path ./data/tripple/tripple_tc_63_3b.tsv  \
                                          --triples_path_eval ./data/tripple/tripple_ct21_63_3b_small.tsv  \
                                         --learning_rate 1e-4 \
                                         --output_model_path ./crossEncoder/models/3b/tc_med_model_63_3b_lr4\
                                         --per_device_train_batch_size 8 \
                                         --gradient_accumulation_steps 16 \
                                         --base_model castorini/monot5-3b-med-msmarco \
                                         --gradient_checkpointing True

### FT tc + medt5
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv  \
#                                         --learning_rate 1e-3 \
#                                         --output_model_path ./crossEncoder/models/lg/tc_medMST5_model\
#                                         --base_model ./crossEncoder/models/lg/medMST5_model \
#                                         --per_device_train_batch_size 4 \
#                                         --gradient_accumulation_steps 32

#srun python finetunet5.py --triples_path triples/triples.train.small.tsv  --save_every_n_steps 10000 --output_model_path monoT5_model
#python ./crossEncoder/finetunet5_tc_val.py --triples_path ./data/tripple/tripple_tc_63_3b.tsv  --triples_path_eval ./data/tripple/tripple_ct21_63_3b_small.tsv  --learning_rate 1e-3 --output_model_path ./crossEncoder/models/3b/tc_med_model_63_3b --per_device_train_batch_size 8 --gradient_accumulation_steps 16