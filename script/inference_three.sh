#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5infer3
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_infer3.txt
#SBATCH -e log/erro_infer3.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# FT medt5 + psuTemp + tc
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/tc_psuTemp_medt5_model \
                                        --outname tc_psuTemp_medt5 \
                                        --field e

# FT medt5 + psuRet + tc
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/tc_psuRet_medT5_model \
                                        --outname tc_psuRet_medt5 \
                                        --field e


