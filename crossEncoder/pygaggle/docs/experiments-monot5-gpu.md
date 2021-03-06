# Experiments on [MS MARCO Passage Retrieval](https://github.com/microsoft/MSMARCO-Passage-Ranking) using monoT5 - Entire Dev Set - mesh with GPUs

This page contains instructions for running monoT5 on the MS MARCO *passage* ranking task with GPUs on Compute Canada Servers.

- monoT5: Document Ranking with a Pretrained Sequence-to-Sequence Model [(Nogueira et al., 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.63.pdf)

Note that there are also separate documents to run MS MARCO ranking tasks on regular GPU. Please see [MS MARCO *document* ranking task](https://github.com/castorini/pygaggle/blob/master/docs/experiments-msmarco-document.md), [MS MARCO *passage* ranking task - Subset](https://github.com/castorini/pygaggle/blob/master/docs/experiments-msmarco-passage-subset.md) and [MS MARCO *passage* ranking task - Entire](https://github.com/castorini/pygaggle/blob/master/docs/experiments-msmarco-passage-entire.md).

Prior to running this, we suggest looking at our first-stage [BM25 ranking instructions](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md).
We rerank the BM25 run files that contain ~1000 passages per query using monoT5.
monoT5 is a pointwise reranker. This means that each document is scored independently using T5.

## Environment Setup
Creat a Python virtual environment for the experiments and install the dependncies

If you haven't installed Anaconda on Compute Canada, please follow this guide [here](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)

```
conda init
conda create --y --name pygaggle python=3.6
conda activate pygaggle
unset PYTHONPATH
conda install -c conda-forge httptools jsonnet --yes
pip install tensorflow-gpu==2.3.0
conda install -c anaconda cudatoolkit=10.1
conda install -c anaconda cudnn
pip install tensorflow-text==2.3.0
git clone https://github.com/google-research/text-to-text-transfer-transformer.git
cd text-to-text-transfer-transformer && git checkout ca1c0627f338927ac753159cb7a1f6caaf2ae19b && pip install --editable . && cd ..
git clone https://github.com/castorini/mesh.git
pip install --editable mesh
```

Also, after setting up the enviroment, go to python interface and import tensorflow to check if cuda can be loaded correctly
```
python
>>> import tensorflow as tf
```
If cuda loaded correctly, output should look something like this: `2021-06-02 09:53:53.794441: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1`
## Data Prep

Since we will use some scripts form PyGaggle to process data and evaluate results, we first install it from source.
```
git clone --recursive https://github.com/castorini/pygaggle.git
cd pygaggle
```

We store all the files in the `data/msmarco_passage` directory.
```
export DATA_DIR=data/msmarco_passage
mkdir ${DATA_DIR}
```

We provide specific data prep instructions for the dev set.

### Dev Set

We download the query, qrels, run and corpus files corresponding to the MS MARCO passage dev set. 

The run file is generated by following the Anserini's [BM25 ranking instructions](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md).

In short, the files are:
- `topics.msmarco-passage.dev-subset.txt`: 6,980 queries from the MS MARCO dev set.
- `qrels.msmarco-passage.dev-subset.txt`: 7,437 pairs of query relevant passage ids from the MS MARCO dev set.
- `run.dev.small.tsv`: Approximately 6,980,000 pairs of dev set queries and retrieved passages using Anserini's BM25.
- `collection.tar.gz`: All passages (8,841,823) in the MS MARCO passage corpus. In this tsv file, the first column is the passage id, and the second is the passage text.

If you are on `Cedar` (a cluster in UWaterloo), these files can be found through at `/projects/rrg-jimmylin/shared_files/gcloud/msmarco/data`

A more detailed description of the data is available [here](https://github.com/castorini/duobert#data-and-trained-models).

If you are not on `Cedar`, you can follow the instructions to download the dev set.

Let's start.
```
cd ${DATA_DIR}
wget https://storage.googleapis.com/duobert_git/run.bm25.dev.small.tsv
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.msmarco-passage.dev-subset.txt
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt
wget https://www.dropbox.com/s/m1n2wf80l1lb9j1/collection.tar.gz
tar -xvf collection.tar.gz
rm collection.tar.gz
mv run.bm25.dev.small.tsv run.dev.small.tsv
cd ../../
```

As a sanity check, we can evaluate the first-stage retrieved documents using the official MS MARCO evaluation script.
```
python tools/scripts/msmarco/msmarco_passage_eval.py ${DATA_DIR}/qrels.msmarco-passage.dev-subset.txt ${DATA_DIR}/run.dev.small.tsv
```

The output should be:
```
#####################
MRR @10: 0.18736452221767383
QueriesRanked: 6980
#####################
```
Then, we prepare the query-doc pairs in the monoT5 input format.
```
python pygaggle/data/create_msmarco_monot5_input.py --queries ${DATA_DIR}/topics.msmarco-passage.dev-subset.txt \
                                      --run ${DATA_DIR}/run.dev.small.tsv \
                                      --corpus ${DATA_DIR}/collection.tsv \
                                      --t5_input ${DATA_DIR}/query_doc_pairs.dev.small.txt \
                                      --t5_input_ids ${DATA_DIR}/query_doc_pair_ids.dev.small.tsv
```
We will get two output files here:
- `query_doc_pairs.dev.small.txt`: The query-doc pairs for monoT5 input.
- `query_doc_pair_ids.dev.small.tsv`: The `query_id`s and `doc_id`s that map to the query-doc pairs. We will use this to map query-doc pairs to their corresponding monoT5 output scores.

The files are made available in our [bucket](https://console.cloud.google.com/storage/browser/castorini/monot5/data).

Note that there might be a memory issue if the monoT5 input file is too large for the memory in the instance. We thus split the input file into multiple files.

```
split --suffix-length 3 --numeric-suffixes --lines 800000 ${DATA_DIR}/query_doc_pairs.dev.small.txt ${DATA_DIR}/query_doc_pairs.dev.small.txt
```

For `query_doc_pairs.dev.small.txt`, we will get 9 files after split. i.e. (`query_doc_pairs.dev.small.txt000` to `query_doc_pairs.dev.small.txt008`).
Note that it is possible that running reranking might still result in OOM issues in which case reduce the number of lines to smaller than `800000`.

## Rerank with monoT5
Let's first define the model type.

monoT5 experiments have 3 model types: base, large, and 3B. To execute the experiments, we need to download model files and config files for each type to `$MODEL_DIR/<base, large, 3B>`.

If you are using `Cedar`, the model checkpoints can be found in this dir `/projects/rrg-jimmylin/shared_files/gcloud/msmarco/monot5/`

The operative config file and model files are also available on google cloud platform. Please use `gsutil cp` to download it. 

If you haven't installed google cloud sdk. please follow this [guide](https://cloud.google.com/sdk/docs/install) here

### Operative config
monoT5-base: [link](https://console.cloud.google.com/storage/browser/_details/t5-data/pretrained_models/base/operative_config.gin)

monoT5-large: [link](https://console.cloud.google.com/storage/browser/_details/t5-data/pretrained_models/large/operative_config.gin)

monoT5-3B: [link](https://console.cloud.google.com/storage/browser/_details/t5-data/pretrained_models/3B/operative_config.gin)

### Model files
monoT5-base: [link](https://console.cloud.google.com/storage/browser/castorini/monot5/experiments/base?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=0&project=neuralresearcher&prefix=&forceOnObjectsSortingFiltering=false)

monoT5-large: [link](https://console.cloud.google.com/storage/browser/castorini/monot5/experiments/large;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=0&project=neuralresearcher&prefix=&forceOnObjectsSortingFiltering=false)

monoT5-3B: [link](https://console.cloud.google.com/storage/browser/castorini/monot5/experiments/3B?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=0&project=neuralresearcher&prefix=&forceOnObjectsSortingFiltering=false)

Create a bash script to request gpus on Compute Canada and run the experiment. You are recommended to use linux editor `vim` to avoid invalid character. 
```
#!/bin/sh
#SBATCH --mem=0
#SBATCH --account=def-jimmylin
#SBATCH --cpus-per-task=32 # request for a whole GPU node
#SBATCH --time=24:0:0
#SBATCH --gres=gpu:v100l:4 # 4 Tesla V100
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

source ~/.bashrc
conda activate pygaggle
cd $HOME/scratch/pygaggle/

export MODEL_DIR=models/monot5/<base, large, 3B>
export CUDA_AVAILABLE_DEVICES=0,1,2,3

for ITER in {000..008}; do
  echo "Running iter: $ITER" >> out.log_eval_exp
  nohup t5_mesh_transformer \
    --model_dir="${MODEL_DIR}" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="infer.gin" \
    --gin_file="beam_search.gin" \
    --gin_param="utils.run.mesh_shape = 'model:1,batch:4'" \ 
    --gin_param="utils.run.mesh_devices = ['gpu:0','gpu:1','gpu:2','gpu:3']" \
    --gin_param="infer_checkpoint_step = 1100000" \
    --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
    --gin_param="Bitransformer.decode.max_decode_length = 2" \
    --gin_param="input_filename = '${DATA_DIR}/query_doc_pairs.dev.small.txt${ITER}'" \
    --gin_param="output_filename = '${DATA_DIR}/query_doc_pair_scores.dev.small.txt${ITER}'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="Bitransformer.decode.beam_size = 1" \
    --gin_param="Bitransformer.decode.temperature = 0.0" \
    --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
    >> out.log_eval_exp 2>&1
done &

tail -100f out.log_eval_exp
```
Then, you can submit the job to Compute Canada with this command `sbatch your_bash_script_name.sh`

With 4 Tesla V100 GPUs, it takes around 9 hours to rerank with monoT5-base, 26 hours with monoT5-large, and 80 hours with monoT5-3B. 
(for 3B, we suggest to request multiple GPU nodes and devide the query files among all the GPU nodes to speed-up the experiment by running in parallel)

## Evaluate reranked results
After reranking is done, we can concatenate all the score files back into one file.
```
cat ${DATA_DIR}/query_doc_pair_scores.dev.small.txt???-1100000 > ${DATA_DIR}/query_doc_pair_scores.dev.small.txt
```

Then we convert the monoT5 output to the required MSMARCO format.
```
python pygaggle/data/convert_monot5_output_to_msmarco_run.py --t5_output ${DATA_DIR}/query_doc_pair_scores.dev.small.txt \
                                                --t5_output_ids ${DATA_DIR}/query_doc_pair_ids.dev.small.tsv \
                                                --mono_run ${DATA_DIR}/run.monot5_${MODEL_NAME}.dev.tsv
```

A sample of the monoT5 outputs and the score files can be found in `/project/rrg-jimmylin/shared_files/gcloud/msmarco/monot5/data/` if you are on `Cedar`.

Now we can evaluate the reranked results using the official MS MARCO evaluation script.
```
python tools/scripts/msmarco/msmarco_passage_eval.py ${DATA_DIR}/qrels.msmarco-passage.dev-subset.txt ${DATA_DIR}/run.monot5_${MODEL_NAME}.dev.tsv
```

In the case of monoT5-3B, the output should be:

```
#####################
MRR @10: 0.39746912948560664
QueriesRanked: 6980
#####################
```
In the case of monoT5-large, the output should be:

```
#####################
MRR @10: 0.39368314231136614
QueriesRanked: 6980
#####################
```
In the case of monoT5-base, the output should be:

```
#####################
MRR @10: 0.3798596329649345
QueriesRanked: 6980
#####################
```
## Trouble Shooting

### GLIBC not found: 

- Tensorflow on GPU reqires sepcific version of gcc compile. Please choose correct one according to [TensorFlow gpu dependency](https://www.tensorflow.org/install/source#gpu). 

```
module list             # Check if you have coreect version of gcc
module spider gcc       # Check available gcc
module load gcc/*.*.*   # Load required gcc
```

### Package not Found:

- If you are using Anaconda to set up virtual environment, please try reinstall the package with `conda install` instead of `pip install`



## Replication Log
+ Results replicated by [@mzzchy](https://github.com/mzzchy) on 2021-09-09 (commit[`c3e21a9`](https://github.com/castorini/pygaggle/commit/c3e21a947d105e0ffc049e1210030b52d6cf9851)) (Compute Canada)
