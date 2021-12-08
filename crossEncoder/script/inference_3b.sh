#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=meshinf3b1
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o log/out_mesh_inf3.txt
#SBATCH -e log/erro_mesh_inf3.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2

#DATA_DIR="data/tripple/tripple_ct21_63_3b_ance.tsv"
#OUT_DIR="data/tripple/tripple_ct21_63_3b_ance_query_doc_pairs.train.tsv"
#srun python ./crossEncoder/pygaggle/pygaggle/data/create_msmarco_monot5_train.py --triples_train $DATA_DIR --output_to_t5 $OUT_DIR
#DATA_DIR="data/tripple/tripple_tc_63_3b_ance.tsv"
#OUT_DIR="data/tripple/tripple_tc_63_3b_ance_query_doc_pairs.train.tsv"
#srun python ./crossEncoder/pygaggle/pygaggle/data/create_msmarco_monot5_train.py --triples_train $DATA_DIR --output_to_t5 $OUT_DIR

MODEL_NAME="3b_med"
MODEL_TYPE="3b"
BASE="/scratch/itee/s4575321/code/synthetic_criteria"
RUN="${BASE}/crossEncoder/mesh_runs/"
MODEL_DIR="${BASE}/crossEncoder/models/${MODEL_TYPE}/med5t"
CKP="1010000"

CONFIG="${MODEL_DIR}/operative_config.gin"
DATA_DIR="${BASE}/crossEncoder/data/ct21_ielab2_monot5_input.tsv"
LOGPATH="${BASE}/log/t5mesh_train.log"
rm out.log_exp
srun
nohup t5_mesh_transformer \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${CONFIG}" \
  --gin_file="infer.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="infer_checkpoint_step = ${CKP}" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="input_filename = '${DATA_DIR}'" \
  --gin_param="output_filename = '${RUN}/${MODEL_NAME}_scores.txt'" \
  --gin_param="utils.run.mesh_shape = 'model:2,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0','gpu:1']" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 8192)" \
  --gin_param="Bitransformer.decode.beam_size = 1" \
  --gin_param="Bitransformer.decode.temperature = 0.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
>> out.log_exp 2>&1 &



