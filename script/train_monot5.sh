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
#srun python ./crossEncoder/finetunet5_val.py --triples_path ../../data/bio-MSmarco_ps/tripple.tsv \
#                                              --triples_path_eval ../../data/bio-MSmarco_ps/tripple_val.tsv  \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/medt5_ps_model \

#sh script/inference_two.sh
#
#srun python src/preprocess/create_pos_neg_trecct.py
#
#### FT tc + medt5
srun python ./crossEncoder/finetunet5_tc_val.py --triples_path data/tripple/tripple_tc_63_3b_ance.tsv  \
                                                --triples_path_eval data/tripple/tripple_ct21_63_3b_ance_small.tsv  \
                                                --learning_rate 1e-4 \
                                                --output_model_path ./crossEncoder/models/t5base/tc_medt5_model/ \
                                                --base_model ./crossEncoder/models/t5base/medt5_ps_model/checkpoint-800

#### FT psutemp train one epoch
#srun python ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_tc_63_base_ance.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/psutemp_medt5_ps_model \
#                                              --base_model ./crossEncoder/models/t5base/medt5_ps_model/checkpoint-800
##
###### FT tc + psutemp
#srun python ./crossEncoder/finetunet5_tc_val.py --triples_path data/tripple/tripple_tc_63_base_ance.tsv  \
#                                                --triples_path_eval data/tripple/tripple_ct21_63_base_ance.tsv  \
#                                                --learning_rate 1e-3 \
#                                                --output_model_path ./crossEncoder/models/t5base/tc_psutemp_medt5_model \
#                                                --base_model ./crossEncoder/models/t5base/psutemp_medt5_ps_model

# inference
path_to_pickle='./data/splits/clean_data_cfg_splits_63_ct21'
path_to_query='../../data/TRECCT2021/topics2021.xml'
path_to_run='crossEncoder/data/ielab-r2.res'
## FT medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/medt5_ps_model/checkpoint-800 \
#                                        --outname base_medt5 \
#                                        --batchsize 128 \
#                                        --path_to_pickle $path_to_pickle \
#                                        --path_to_query $path_to_query \
#                                        --path_to_run $path_to_run

### FT tc + medt5
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/tc_medt5_model \
                                        --outname base_tc_medt5 \
                                        --batchsize 128 \
                                        --path_to_pickle $path_to_pickle \
                                        --path_to_query $path_to_query \
                                        --path_to_run $path_to_run

#### FT psutemp + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/psutemp_medt5_ps_model \
#                                        --outname base_psutemp_medt5 \
#                                        --batchsize 128 \
#                                        --path_to_pickle $path_to_pickle \
#                                        --path_to_query $path_to_query \
#                                        --path_to_run $path_to_run
#
#### FT psutemp + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/tc_psutemp_medt5_model \
#                                        --outname base_tc_psutemp_medt5 \
#                                        --batchsize 128 \
#                                        --path_to_pickle $path_to_pickle \
#                                        --path_to_query $path_to_query \
#                                        --path_to_run $path_to_run

#srun python crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv  --save_every_n_steps 10000 --output_model_path test_monoT5_model