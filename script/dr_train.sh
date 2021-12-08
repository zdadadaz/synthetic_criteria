#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=drtrain
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -o log/out_drtrain.txt
#SBATCH -e log/erro_drtrain.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-smx2:2

# preprocess
srun python denseRetrieve/preprocess/prepare_tc.py

## FT bio
srun python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 29015 \
    -m tevatron.driver.train \
    --output_dir ./denseRetrieve/models/ance/biot5 \
    --model_name_or_path castorini/ance-msmarco-passage \
    --do_train \
    --save_steps 5000 \
    --train_dir ./denseRetrieve/data/bio_trainset \
    --fp16 \
    --per_device_train_batch_size 20 \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --train_n_passages 8 \
    --q_max_len 512 \
    --p_max_len 512 \
    --negatives_x_device \
    --cache_dir ./denseRetrieve/cache \
    --dataloader_num_workers 10 \
    --add_pooler \
    --grad_cache \
    --overwrite_output_dir

## psu temp
#srun python -m torch.distributed.launch \
#    --nproc_per_node=2 \
#    --master_port 29015 \
#    -m tevatron.driver.train \
#    --output_dir ./denseRetrieve/models/ance/psuTemp \
#    --model_name_or_path castorini/ance-msmarco-passage \
#    --do_train \
#    --save_steps 10000 \
#    --train_dir ./denseRetrieve/data/psu_temp_trainset \
#    --fp16 \
#    --per_device_train_batch_size 20 \
#    --learning_rate 1e-6 \
#    --num_train_epochs 50 \
#    --train_n_passages 8 \
#    --q_max_len 512 \
#    --p_max_len 512 \
#    --negatives_x_device \
#    --cache_dir ./denseRetrieve/cache \
#    --dataloader_num_workers 10 \
#    --add_pooler \
#    --grad_cache \
#    --overwrite_output_dir



### bio + tc
#srun python -m torch.distributed.launch \
#    --nproc_per_node=2 \
#    --master_port 29015 \
#    -m tevatron.driver.train \
#    --output_dir ./denseRetrieve/models/ance/tc_biot5 \
#    --model_name_or_path castorini/ance-msmarco-passage \
#    --target_model_path ./denseRetrieve/models/ance/biot5 \
#    --do_train \
#    --save_steps 1000 \
#    --train_dir ./denseRetrieve/data/tc_trainset \
#    --fp16 \
#    --per_device_train_batch_size 20 \
#    --learning_rate 1e-6 \
#    --num_train_epochs 200 \
#    --train_n_passages 8 \
#    --q_max_len 512 \
#    --p_max_len 512 \
#    --negatives_x_device \
#    --cache_dir ./denseRetrieve/cache \
#    --dataloader_num_workers 10 \
#    --add_pooler \
#    --grad_cache \
#    --overwrite_output_dir
#
## psu temp + tc
#srun python -m torch.distributed.launch \
#    --nproc_per_node=2 \
#    --master_port 29015 \
#    -m tevatron.driver.train \
#    --output_dir ./denseRetrieve/models/ance/tc_psuTemp \
#    --model_name_or_path castorini/ance-msmarco-passage \
#    --target_model_path ./denseRetrieve/models/ance/psuTemp/checkpoint-10000 \
#    --do_train \
#    --save_steps 1000 \
#    --train_dir ./denseRetrieve/data/tc_trainset \
#    --fp16 \
#    --per_device_train_batch_size 20 \
#    --learning_rate 1e-6 \
#    --num_train_epochs 200 \
#    --train_n_passages 8 \
#    --q_max_len 512 \
#    --p_max_len 512 \
#    --negatives_x_device \
#    --cache_dir ./denseRetrieve/cache \
#    --dataloader_num_workers 10 \
#    --add_pooler \
#    --grad_cache \
#    --overwrite_output_dir
#
#
## psu ret
#srun python -m torch.distributed.launch \
#    --nproc_per_node=2 \
#    --master_port 29015 \
#    -m tevatron.driver.train \
#    --output_dir ./denseRetrieve/models/ance/psuRet \
#    --model_name_or_path castorini/ance-msmarco-passage \
#    --do_train \
#    --save_steps 5000 \
#    --train_dir ./denseRetrieve/data/psu_ret_trainset \
#    --fp16 \
#    --per_device_train_batch_size 20 \
#    --learning_rate 1e-6 \
#    --num_train_epochs 20 \
#    --train_n_passages 8 \
#    --q_max_len 512 \
#    --p_max_len 512 \
#    --negatives_x_device \
#    --cache_dir ./denseRetrieve/cache \
#    --dataloader_num_workers 10 \
#    --add_pooler \
#    --grad_cache \
#    --overwrite_output_dir

# psu ret + tc
#srun python -m torch.distributed.launch \
#    --nproc_per_node=2 \
#    --master_port 29015 \
#    -m tevatron.driver.train \
#    --output_dir ./denseRetrieve/models/ance/tc_psuRet \
#    --model_name_or_path castorini/ance-msmarco-passage \
#    --target_model_path ./denseRetrieve/models/ance/psuRet \
#    --do_train \
#    --save_steps 1000 \
#    --train_dir ./denseRetrieve/data/tc_trainset \
#    --fp16 \
#    --per_device_train_batch_size 20 \
#    --learning_rate 1e-6 \
#    --num_train_epochs 200 \
#    --train_n_passages 8 \
#    --q_max_len 512 \
#    --p_max_len 512 \
#    --negatives_x_device \
#    --cache_dir ./denseRetrieve/cache \
#    --dataloader_num_workers 10 \
#    --add_pooler \
#    --grad_cache \
#    --overwrite_output_dir



#python -m tevatron.driver.train --output_dir ./denseRetrieve/models/ance/tc_biot5 --target_model_path ./denseRetrieve/models/ance/biot5 --model_name_or_path castorini/ance-msmarco-passage --do_train --save_steps 5000 --train_dir ./denseRetrieve/data/tc_trainset --fp16 --per_device_train_batch_size 20 --learning_rate 1e-4  --num_train_epochs 20 --train_n_passages 4 --cache_dir ./denseRetrieve/cache --add_pooler --overwrite_output_dir