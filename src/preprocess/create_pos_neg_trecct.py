import sys

if 'src/' not in sys.path:
    sys.path.append('src/')
from utils import readfile as rf
import pickle
from tqdm import tqdm
import re
import os
import numpy as np

type2tname = {'e': 'eligibility', 'd': 'desc'}
map2ft = {'eligibility': 'eligibility', 'desc': 'description'}

def get_strong_samples(strong_segment, trial, out_samples):
    title = trial['title'] if 'title' in trial else 'NA'
    cond = trial['condition'] if 'condition' in trial else 'NA'
    p_list = []
    for dtype in ['e', 'd']:
        if strong_segment[dtype]:
            if strong_segment[dtype][0] == -1:
                out_samples.append(f"title: {title} condition: {cond} {map2ft[type2tname[dtype]]}: NA")
                p_list.append("NA")
            else:
                pidx = strong_segment[dtype][0]
                p = trial[type2tname[dtype]][pidx]
                out_samples.append(f"title: {title} condition: {cond} {map2ft[type2tname[dtype]]}: {p}")
                p_list.append(p)
    if len(p_list) == 2:  # ed
        out_samples.append(f"title: {title} condition: {cond} eligibility: {p_list[0]} description: {p_list[1]}")


def get_weak_samples(strong_segment, trial, out_samples):
    title = trial['title'] if 'title' in trial else 'NA'
    cond = trial['condition'] if 'condition' in trial else 'NA'
    p_list = [[],[]]
    for idx, etype in enumerate(['eligibility', 'desc']):
        if etype in trial and trial[etype]:
            for idxp, p in enumerate(trial[etype]):
                # check 'not None' and 'not equal to strong segment'
                if p and strong_segment[etype[0]] and strong_segment[etype[0]][0] != -1 and idxp != \
                        strong_segment[etype[0]][0]:
                    out_samples.append(f"title: {title} condition: {cond} {map2ft[etype]}: {p}")
                    p_list[idx].append(p)
    if p_list[0] and p_list[1]:  # ed
        for i in range(min(len(p_list[0]), len(p_list[1]))):
            pos_sample = np.random.choice(p_list[0], replace=False)
            des_sample = np.random.choice(p_list[1], replace=False)
            out_samples.append(f"title: {title} condition: {cond} eligibility: {pos_sample} description: {des_sample}")


def random_sample_strong_weak_negatives(pos_n, out_samples, strong_samples, weak_samples):
    weak_n = int(pos_n * 15/60)
    strong_n = pos_n - weak_n
    for _ in range(strong_n):
        neg_strong = np.random.choice(strong_samples, replace=True)
        out_samples.append(neg_strong)
    for _ in range(weak_n):
        neg_strong = np.random.choice(weak_samples, replace=True)
        out_samples.append(neg_strong)


def create_pos_neg(path_to_file, path_to_pickle, query, log_path, outname):
    qrels = rf.get_all_judged(path_to_file, 0)
    strong_segments = rf.read_log(log_path)
    trials = pickle.load(open(path_to_pickle, 'rb'))
    nctid2idx = {i['number']: idx for idx, i in enumerate(trials)}
    qid_list = list(query.keys())
    out_path = os.path.join('./data/tripple', outname)
    if os.path.exists(out_path):
        os.system('rm {}'.format(out_path))
    fout = open(out_path, 'w')
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
        pos_samples = pos_samples * 20
        pos_n = len(pos_samples)
        random_sample_strong_weak_negatives(pos_n, neg_samples, neg_strong_samples, neg_weak_samples)
        for pos, neg in zip(pos_samples, neg_samples):
            out_text = f"{text}\t{pos}\t{neg}\n"
            out_text = re.sub(r'\r|\n|\t|\s\s+', ' ', out_text)
            fout.write(out_text)
        fout.flush()
    fout.close()


if __name__ == '__main__':
    np.random.seed(123)
    path_to_file = '../../data/test_collection/qrels-clinical_trials.tsv'
    path_to_pickle = './data/splits/clean_data_cfg_splits_63'
    path_to_query = '../../data/test_collection/topics-2014_2015-description.topics'
    log_path = 'crossEncoder/data/ct2016_e_individual_pscore.log'
    query = rf.read_ts_topic(path_to_query)
    outname = 'tripple_tc_63_3b_ance.tsv'
    print(outname)
    create_pos_neg(path_to_file, path_to_pickle, query, log_path, outname)
