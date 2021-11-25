#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5infer3
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o log/out_infer3.txt
#SBATCH -e log/erro_infer3.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# bio
srun python ./crossEncoder/inference.py --base_model ./crossEncoder/models/t5base/medMST5_ps_model \
                                        --outname medt5 \
                                        --log_path crossEncoder/runs/medt5_e_individual_pscore.log

# tc
srun python ./crossEncoder/inference.py --base_model ./crossEncoder/models/t5base/tc_model \
                                        --outname tc_model \
                                        --log_path crossEncoder/runs/tc_model_e_individual_pscore.log

# psu temp
srun python ./crossEncoder/inference.py --base_model ./crossEncoder/models/t5base/psuTempT5_model \
                                        --outname psuTemp \
                                        --log_path crossEncoder/runs/psuTemp_e_individual_pscore.log

# tc + psu temp
srun python ./crossEncoder/inference.py --base_model ./crossEncoder/models/t5base/tc_psuTempT5_model \
                                        --outname tc_psuTempT5 \
                                        --log_path crossEncoder/runs/tc_psuTempT5_e_individual_pscore.log

# tc + bio
srun python ./crossEncoder/inference.py --base_model ./crossEncoder/models/t5base/tc_medMST5_model \
                                        --outname tc_medt5 \
                                        --log_path crossEncoder/runs/tc_medt5_e_individual_pscore.log