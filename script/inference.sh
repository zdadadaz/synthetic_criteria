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
#SBATCH --gres=gpu:tesla-smx2:1

# FT medt5
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/medMST5_ps_model \
                                        --outname medt5 \

# FT psu temp
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/psuTempT5_model \
                                        --outname psuTemp \

# FT psu ret
#srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/psuRetT5_model \
#                                        --outname psuRet \

# FT tc
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/tc_model \
                                        --outname tc_model \


#python -m pdb ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/t5base/tc_medMST5_model --outname tc_medt5_test