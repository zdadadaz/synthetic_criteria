import os
import random


def prepare_tripple():
    random.seed(123)
    path_to_dir = '../../data/bio-MSmarco_ps/'
    queries = {}
    for line in open(os.path.join(path_to_dir, 'queries.train.tsv')):
        doc_id, text = line.strip().split('\t')
        queries[doc_id] = text
    collections = {}
    for line in open(os.path.join(path_to_dir, 'collection.tsv')):
        doc_id, text = line.strip().split('\t')
        collections[doc_id] = text

    all_collection = list(collections.keys())
    with open(os.path.join(path_to_dir, 'qrels.train.tsv')) as fin:
        lines = fin.readlines()
    with open(path_to_dir + 'tripple.tsv', 'w') as f:
        for lidx in range(int(len(lines) * 0.8)):
            line = lines[lidx]
            qid, _, doc_id, rel = line.strip().split('\t')
            negdoc = random.choice(all_collection)
            while doc_id == negdoc:
                negdoc = random.choice(all_collection)
            f.write("{}\t{}\t{}\n".format(queries[qid], collections[doc_id], collections[negdoc]))
            f.flush()
    with open(path_to_dir + 'tripple_val.tsv', 'w') as f:
        for lidx in range(int(len(lines) * 0.8) + 1, len(lines)):
            line = lines[lidx]
            qid, _, doc_id, rel = line.strip().split('\t')
            negdoc = random.choice(all_collection)
            while doc_id == negdoc:
                negdoc = random.choice(all_collection)
            f.write("{}\t{}\t{}\n".format(queries[qid], collections[doc_id], collections[negdoc]))
            f.flush()


if __name__ == '__main__':
    prepare_tripple()
