#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=runOne
#SBATCH -n 1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o log/out2.txt
#SBATCH -e log/erro2.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2

#srun python  pyserini/scripts/trec-ct/convert_trec21_ct_to_json.py --input_dir ../../data/TRECCT2021/clinicaltrials_xml/ --output_dir ../../data/TRECCT2021/pyserini_json/
#srun python pyserini/scripts/trec-ct/convert_topic_xml_to_tsv.py --topics ../../data/TRECCT2021/topics2021.xml --queries ../../data/TRECCT2021/ctqueries2021.tsv
#srun python -m pyserini.index --collection JsonCollection --generator DefaultLuceneDocumentGenerator --threads 9 --input ../../data/TRECCT2021/pyserini_json/ --index ../pyserini/indexes/lucene-index-ct-2021  --storePositions --storeDocvectors --storeRaw
#srun python -m pyserini.search --topics ../../data/TRECCT2021/ctqueries2021.tsv --index ../pyserini/indexes/lucene-index-ct-2021/ --output sparseRetrieve/runs/bm25_wl.res --hits 1000 --bm25 --k1 0.9 --b 0.4
#srun python -m pyserini.search --topics ../../data/TRECCT2021/ctqueries2021.tsv --index ../pyserini/indexes/lucene-index-ct-2021/ --output sparseRetrieve/runs/bm25_rm3_wl.res --hits 1000 --bm25 --rm3 --k1 0.9 --b 0.4

#srun python src/gen_mimicIII.py
#srun python src/preprocess/prepare_data.py
#srun python src/preprocess/prepare_biomsmarco.py
#srun python src/preprocess/create_pos_neg.py
#srun python denseRetrieve/preprocess/tokenize_collection.py
