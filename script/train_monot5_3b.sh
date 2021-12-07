#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5_3b
#SBATCH -n 1
#SBATCH --time=60:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH -o log/out_3b.txt
#SBATCH -e log/erro_3b.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla-smx2:2


srun python ./crossEncoder/finetunet5_tc_val.py --triples_path data/tripple/tripple_tc_63_3b_ance.tsv  \
                                          --triples_path_eval data/tripple/tripple_ct21_63_3b_ance.tsv  \
                                         --learning_rate 1e-3 \
                                         --output_model_path ./crossEncoder/models/3b/tc_med_model_63_3b_ance \
                                         --per_device_train_batch_size 16 \
                                         --gradient_accumulation_steps 8 \
                                         --base_model castorini/monot5-3b-med-msmarco \
                                         --gradient_checkpointing True \
                                         --model_parallel 1

### inference
outname='3b_ct_medt5_ance'
base_model='./crossEncoder/models/3b/tc_med_model_63_3b_ance'
path_to_pickle='./data/splits/clean_data_cfg_splits_63_ct21'
path_to_query='../../data/TRECCT2021/topics2021.xml'
path_to_run='crossEncoder/data/ielab-r2.res'
## FT tc + medt5 e
srun python ./crossEncoder/inference_e.py --base_model $base_model \
                                        --outname $outname \
                                        --batchsize 64 \
                                        --model_parallel 1 \
                                        --path_to_pickle $path_to_pickle \
                                        --path_to_query $path_to_query \
                                        --path_to_run $path_to_run

#srun python finetunet5.py --triples_path triples/triples.train.small.tsv  --save_every_n_steps 10000 --output_model_path monoT5_model
#python -m pdb ./crossEncoder/finetunet5_tc_val.py --triples_path ./data/tripple/tripple_tc_63_3b_ance.tsv  --triples_path_eval ./data/tripple/tripple_ct21_63_3b_ance.tsv  --learning_rate 1e-3 --output_model_path ./crossEncoder/models/3b/tc_med_model_63_3b --per_device_train_batch_size 8 --gradient_accumulation_steps 16