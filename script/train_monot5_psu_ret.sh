#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=psu_2
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_psu2.txt
#SBATCH -e log/erro_psu2.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla-smx2:1


# FT psu ret
srun python ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_psu_ret_split.tsv \
                                              --learning_rate 1e-3 \
                                              --output_model_path ./crossEncoder/models/t5base/psuRetT5_model

## FT psu ret + medt5
#srun python ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_psu_ret.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/psuRet_medT5_model \
#                                              --base_model ./crossEncoder/medMST5_ps_model
#
### FT tc + psu ret
srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
                                              --learning_rate 1e-3 \
                                              --output_model_path ./crossEncoder/models/t5base/tc_psuRet_model \
                                              --base_model ./crossEncoder/models/t5base/psuRetT5_model

#srun python -m pdb ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_psu_ret_split.tsv --learning_rate 1e-3 --output_model_path ./crossEncoder/models/t5base/psuRetT5_model