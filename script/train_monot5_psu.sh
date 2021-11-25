#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=psu_1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH -o log/out_psu1.txt
#SBATCH -e log/erro_psu1.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla-smx2:1


# FT psu temp
#srun python ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_psu_temp_split.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/psuTempT5_model

## FT psu temp with medt5
#srun python ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_psu_temp.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/psuTemp_medT5_model \
#                                              --base_model ./crossEncoder/medMST5_ps_model
#
#### FT tc with psu temp
#srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
#                                              --learning_rate 1e-3 \
#                                              --output_model_path ./crossEncoder/models/t5base/tc_psuTemp_model \
#                                              --base_model ./crossEncoder/models/t5base/psuTempT5_model


#####  large
# FT psu temp
srun python ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_psu_temp_split.tsv \
                                              --learning_rate 1e-3 \
                                              --output_model_path ./crossEncoder/models/lg/psuTempT5_model \
                                              --base_model castorini/monot5-large-msmarco-10k

### FT tc with psu temp
srun python ./crossEncoder/finetunet5_tc.py --triples_path ./data/tripple/tripple_tc.tsv \
                                              --learning_rate 1e-3 \
                                              --output_model_path ./crossEncoder/models/lg/tc_psuTemp_model \
                                              --base_model ./crossEncoder/models/lg/psuTempT5_model

# inference
# FT psu temp
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/lg/psuTempT5_model \
                                        --outname psuTemp_lg

# FT tc + psu temp
srun python ./crossEncoder/inference_e.py --base_model ./crossEncoder/models/lg/tc_psuTemp_model \
                                        --outname tc_psuTemp_lg


#python ./crossEncoder/finetunet5.py --triples_path ./data/tripple/tripple_psu_temp.tsv --learning_rate 1e-3 --output_model_path ./crossEncoder/test_psuTempT5_model