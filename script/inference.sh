#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5infer
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_infer1.txt
#SBATCH -e log/erro_infer1.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# FT medt5
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/medMST5_ps_model \
                                        --outname medt5 \
                                        --field e

# FT psu temp
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/psuTempT5_model \
                                        --outname psuTemp \
                                        --field e

# FT psu ret
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/psuRetT5_model \
                                        --outname psuRet \
                                        --field e

# FT tc ret
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/tc_model \
                                        --outname tc_model \
                                        --field e


#python ./crossEncoder/inference_e.py --base_model ./crossEncoder/medMST5_ps_model --outname medt5_test --field e