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
#SBATCH --gres=gpu:1


## FT psu temp + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/psuTemp_medT5_model \
#                                        --outname psuTemp_medt5 \
#                                        --field e
#
## FT psu ret + medt5
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/psuRet_medT5_model \
#                                        --outname psuRet_medt5 \
#                                        --field e

# FT tc + medt5
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/tc_medMST5_model \
                                        --outname tc_medt5 \
                                        --field e
# FT tc + psuRet
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/tc_psuRetT5_model \
                                        --outname tc_psuRetT5 \
                                        --field e

# FT tc + psuTemp
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/tc_psuTempT5_model \
                                        --outname tc_psuTempT5 \
                                        --field e

