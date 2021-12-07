#!/bin/bash

outname='ct2016_base'
base_model='crossEncoder/models/t5base/medt5_ps_model/checkpoint-800'
path_to_pickle='./data/splits/clean_data_cfg_splits_63'
path_to_query='../../data/test_collection/topics-2014_2015-description.topics'
path_to_run='data/judgment/ct2016_judgement.res'
## FT tc + medt5 e
srun python ./crossEncoder/inference_e.py --base_model $base_model \
                                        --outname $outname \
                                        --batchsize 32 \
                                        --path_to_pickle $path_to_pickle \
                                        --path_to_query $path_to_query \
                                        --path_to_run $path_to_run

outname='ct2021_base'
base_model='crossEncoder/models/t5base/medt5_ps_model/checkpoint-800'
path_to_pickle='./data/splits/clean_data_cfg_splits_63_ct21'
path_to_query='../../data/TRECCT2021/topics2021.xml'
path_to_run='data/judgment/ct2021_judgement.res'
srun python ./crossEncoder/inference_e.py --base_model $base_model \
                                        --outname $outname \
                                        --batchsize 32 \
                                        --path_to_pickle $path_to_pickle \
                                        --path_to_query $path_to_query \
                                        --path_to_run $path_to_run