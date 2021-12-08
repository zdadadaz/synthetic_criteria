#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=mesh3b
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o log/out_mesh3b_train1.txt
#SBATCH -e log/erro_mesh3b_train1.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4

#DATA_DIR="data/tripple/tripple_ct21_63_3b_ance.tsv"
#OUT_DIR="data/tripple/tripple_ct21_63_3b_ance_query_doc_pairs.train.tsv"
#srun python ./crossEncoder/pygaggle/pygaggle/data/create_msmarco_monot5_train.py --triples_train $DATA_DIR --output_to_t5 $OUT_DIR
#DATA_DIR="data/tripple/tripple_tc_63_3b_ance.tsv"
#OUT_DIR="data/tripple/tripple_tc_63_3b_ance_query_doc_pairs.train.tsv"
#srun python ./crossEncoder/pygaggle/pygaggle/data/create_msmarco_monot5_train.py --triples_train $DATA_DIR --output_to_t5 $OUT_DIR

BASE="/scratch/itee/s4575321/code/synthetic_criteria"
MODEL_NAME="3b"
MODEL_DIR="${BASE}/crossEncoder/models/${MODEL_NAME}/med5t/model.ckpt-1010000"
DATA_DIR="${BASE}/data/tripple/tripple_tc_63_3b_ance_query_doc_pairs.train.tsv"
CONFIG="${BASE}/crossEncoder/models/${MODEL_NAME}/med5t/operative_config.gin"
OUTPATH="${BASE}/crossEncoder/models/${MODEL_NAME}/tc_med_mesh"

mkdir -p $OUTPATH
#rm out.log_exp
# srun
nohup t5_mesh_transformer \
  --model_dir="${OUTPATH}" \
  --gin_file="${CONFIG}" \
  --gin_param="init_checkpoint = '${MODEL_DIR}'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1_3B.gin" \
  --gin_param="utils.run.mesh_shape = 'model:2,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0','gpu:1']" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = '${DATA_DIR}'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1011000" \
  --gin_param="run.save_checkpoints_steps = 1000" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
  --gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 512" \
  >> out.log_exp 2>&1 &



