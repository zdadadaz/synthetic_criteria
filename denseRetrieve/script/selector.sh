#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=drtk
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_dr_tokenize.txt
#SBATCH -e log/erro_dr_tokenize.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

srun python denseRetrieve/selector.py --base_model 'denseRetrieve/models/ance/biot5' \
                                --outname 'bio_test' \
                                --query_reps 'denseRetrieve/data/ct21/ance_encoding/biot5/qry.pt' \
                                --passage_reps 'denseRetrieve/data/ct21/ance_encoding/biot5'


#python denseRetrieve/selector.py --base_model 'denseRetrieve/models/ance/biot5' --outname 'bio_test' --query_reps 'denseRetrieve/data/ct21/ance_encoding/biot5/qry.pt' --passage_reps 'denseRetrieve/data/ct21/ance_encoding/biot5'
