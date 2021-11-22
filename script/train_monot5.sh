#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out1.txt
#SBATCH -e log/erro1.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# FT medt5
#srun python ./crossEncoder/finetunet5.py --triples_path ../../data/bio-MSmarco_ps/tripple.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/medMST5_ps_model

### FT tc
srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
                                              --learning_rate 1e-3 \
                                              --output_model_path ./crossEncoder/models/t5base/tc_model


# FT tc + medt5
srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
                                              --learning_rate 1e-3 \
                                              --output_model_path ./crossEncoder/models/t5base/tc_medMST5_model \
                                              --base_model ./crossEncoder/models/t5base/medMST5_ps_model

############## train T5 3B
## FT tc
#srun python ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_tc.tsv  \
#                                         --learning_rate 1e-3 \
#                                         --output_model_path ./crossEncoder/models/3b/tc_medMST5_3B_model\
#                                         --base_model castorini/monot5-3b-med-msmarco



#srun python -m pdb crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv  --save_every_n_steps 10000 --output_model_path test_monoT5_model