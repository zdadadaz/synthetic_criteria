#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=runTwo
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out1.txt
#SBATCH -e log/erro1.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

srun python src/split_note.py
#srun python src/gen_mimicIII.py
#srun python src/preprocess/prepare_data.py
#srun python src/preprocess/prepare_biomsmarco.py
#srun python src/preprocess/create_pos_neg.py