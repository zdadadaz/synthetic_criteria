#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5inf3b3
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o log/out_infer_3b3.txt
#SBATCH -e log/erro_infer_3b3.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:2


## FT psu temp + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/psuTemp_medT5_model \
#                                        --outname psuTemp_medt5 \
#
## FT psu ret + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/psuRet_medT5_model \
#                                        --outname psuRet_medt5 \


outname='ct_medt5_lr4'
base_model='crossEncoder/models/3b/tc_med_model_63_3b_lr4/checkpoint-126'
path_to_pickle='./data/splits/clean_data_cfg_splits_63_ct21'
path_to_run='data/judgment/ct2016_judgement.res'
## FT tc + medt5 e
srun python ./crossEncoder/inference_e.py --base_model $base_model \
                                        --outname $outname \
                                        --batchsize 32 \
                                        --model_parallel 1 \
                                        --path_to_pickle $path_to_pickle \
                                        --path_to_run $path_to_run

## FT tc + medt5 ed
#srun python ./crossEncoder/inference.py --base_model $base_model \
#                                        --outname $outname \
#                                        --batchsize 16 \
#                                        --log_path crossEncoder/runs/${outname}_e_individual_pscore.log \
#                                        --model_parallel 1