import sys

if 'src/' not in sys.path:
    sys.path.append('src/')
from utils import readfile as rf
import pickle
from tqdm import tqdm
import re
import os
import numpy as np
from create_pos_neg_trecct import (get_strong_samples, get_weak_samples, random_sample_strong_weak_negatives)

type2tname = {'e': 'eligibility', 'd': 'desc'}
map2ft = {'eligibility': 'eligibility', 'desc': 'description'}

def random_sample_strong_weak_negatives(pos_n, out_samples, strong_samples, weak_samples):
    weak_n = int(pos_n * 15/60)
    strong_n = pos_n - weak_n
    for _ in range(strong_n):
        neg_strong = np.random.choice(strong_samples, replace=True)
        out_samples.append(neg_strong)
    for _ in range(weak_n):
        neg_strong = np.random.choice(weak_samples, replace=True)
        out_samples.append(neg_strong)


def create_pos_neg(path_to_file, path_to_pickle, query, log_path, outname, n_times):
    qrels = rf.get_all_judged(path_to_file, 0)
    strong_segments = rf.read_log(log_path)
    trials = pickle.load(open(path_to_pickle, 'rb'))
    nctid2idx = {i['number']: idx for idx, i in enumerate(trials)}
    qid_list = list(query.keys())
    out_path = os.path.join('./data/tripple', outname)
    if os.path.exists(out_path):
        os.system('rm {}'.format(out_path))
    with open(out_path, 'a+') as fout:
        for q_idx in tqdm(range(len(query))):
            qid = qid_list[q_idx]
            text = query[qid]
            text = re.sub(r'\r|\n|\t|\s\s+', ' ', text)
            if qid not in qrels:
                continue
            pos_samples = []
            neg_samples = []
            neg_strong_samples = []
            neg_weak_samples = []
            for docid in qrels[qid]['pos']:  # positive samples
                # positive repeat 20 times
                get_strong_samples(strong_segments[qid][docid], trials[nctid2idx[docid]], pos_samples)
            for docid in qrels[qid]['neg']:  # negative samples
                # strong samples
                get_strong_samples(strong_segments[qid][docid], trials[nctid2idx[docid]], neg_strong_samples)
                # weak samples
                get_weak_samples(strong_segments[qid][docid], trials[nctid2idx[docid]], neg_weak_samples)
            pos_samples = pos_samples * n_times
            pos_n = len(pos_samples)
            random_sample_strong_weak_negatives(pos_n, neg_samples, neg_strong_samples, neg_weak_samples)
            for pos, neg in zip(pos_samples, neg_samples):
                pos = re.sub(r'\r|\n|\t|\s\s+', ' ', pos)
                neg = re.sub(r'\r|\n|\t|\s\s+', ' ', neg)
                out_text = f"{text}\t{pos}\t{neg}\n"
                fout.write(out_text)
            fout.flush()

if __name__ == '__main__':
    np.random.seed(123)
    path_to_file = '../../data/test_collection/qrels-clinical_trials.tsv'
    path_to_pickle = './data/splits/clean_data_cfg_splits_63'
    path_to_query = '../../data/test_collection/topics-2014_2015-description.topics'
    # log_path = 'crossEncoder/data/ct2016_e_individual_pscore.log'
    log_path = 'crossEncoder/runs/ct2016_base_e_individual_pscore.log'
    query = rf.read_ts_topic(path_to_query)
    outname = 'tripple_tc_63_base_ance.tsv'
    print(outname)
    create_pos_neg(path_to_file, path_to_pickle, query, log_path, outname, 20)

    path_to_file = '../../data/TRECCT2021/trec_2021_binarized_qrels.txt'
    path_to_pickle = './data/splits/clean_data_cfg_splits_63_ct21'
    path_to_query = '../../data/TRECCT2021/topics2021.xml'
    # log_path = 'crossEncoder/data/ct2021_e_individual_pscore.log'
    log_path = 'crossEncoder/runs/ct2021_base_e_individual_pscore.log'
    query = rf.read_topics_ct21(path_to_query)
    outname = 'tripple_ct21_63_base_ance.tsv'
    print(outname)
    create_pos_neg(path_to_file, path_to_pickle, query, log_path, outname, 1)