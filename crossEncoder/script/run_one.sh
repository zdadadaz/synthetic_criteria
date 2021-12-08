#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=r11
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_11.txt
#SBATCH -e log/erro_11.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2

Query='../../data/TRECCT2021/topics2021.xml'
Run='crossEncoder/data/ielab-r2.res'
Pickle='./data/splits/clean_data_cfg_splits_63_ct21'
t5_input='crossEncoder/data/ct21_ielab2_monot5_input.tsv'
t5_input_ids='crossEncoder/data/ct21_ielab2_monot5_input.ids.tsv'

srun python crossEncoder/preprocess/create_monot5_input.py --queries $Query \
                                                    --run $Run \
                                                    --path_to_pickle $Pickle \
                                                    --t5_input $t5_input \
                                                    --t5_input_ids $t5_input_ids

