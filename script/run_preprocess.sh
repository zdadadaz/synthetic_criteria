#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=preproc
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH -o log/out3.txt
#SBATCH -e log/erro3.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

#srun python src/gen_mimicIII.py
srun python src/preprocess/prepare_data.py
srun python src/preprocess/create_pos_neg.py
#srun python denseRetrieve/preprocess/prepare_tc.py