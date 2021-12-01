#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5infer2
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_infer2.txt
#SBATCH -e log/erro_infer2.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla-smx2:1


## FT psu temp + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/psuTemp_medT5_model \
#                                        --outname psuTemp_medt5
#
## FT psu ret + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/psuRet_medT5_model \
#                                        --outname psuRet_medt5

## FT tc + medt5
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/tc_medMST5_model \
                                        --outname tc_medt5 \
                                        --batchsize 128

srun python ./crossEncoder/inference.py --base_model ./crossEncoder/models/t5base/tc_medMST5_model \
                                        --outname tc_medt5 \
                                        --batchsize 128 \
                                        --log_path crossEncoder/runs/tc_medt5_e_individual_pscore.log

## FT tc + psuRet
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/tc_psuRetT5_model \
#                                        --outname tc_psuRetT5

# FT tc + psuTemp
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/tc_psuTempT5_model \
#                                        --outname tc_psuTempT5

