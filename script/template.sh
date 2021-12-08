#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=t5infer2
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out_infer2.txt
#SBATCH -e log/erro_infer2.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla-smx2:1