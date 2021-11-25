#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5_tc
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_tc.txt
#SBATCH -e log/erro_tc.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# FT medt5
#srun python ./crossEncoder/finetunet5.py --triples_path ../../data/bio-MSmarco_ps/tripple.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/medMST5_ps_model


# FT tc + psu temp
srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
                                              --learning_rate 1e-3 \
                                              --output_model_path ./crossEncoder/models/t5base/tc_psuTempT5_model \
                                              --base_model ./crossEncoder/models/t5base/psuTempT5_model

# FT tc + psu ret
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/tc_psuRetT5_model \
#                                              --base_model ./crossEncoder/models/t5base/psuRetT5_model

## FT tc + psu ret + medt5
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/tc_psuRet_medT5_model \
#                                              --base_model ./crossEncoder/models/t5base/psuRet_medT5_model
#
## FT tc + psu temp + medt5
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/tc_psuTemp_medT5_model \
#                                              --base_model ./crossEncoder/models/t5base/psuTemp_medT5_model



#srun python finetunet5.py --triples_path triples/triples.train.small.tsv  --save_every_n_steps 10000 --output_model_path monoT5_model