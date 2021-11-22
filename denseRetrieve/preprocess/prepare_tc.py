from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5, T5BatchTokenizer
import sys
import json

if 'src/' not in sys.path:
    sys.path.append('src/')
from utils import readfile as rf
import pickle
import random
import os
import re
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import torch


def to_t5_input(sample):
    if sample[0] == 'i' or sample[0] == 'e':
        text = f'title: {sample[1]} condition: {sample[2]} eligibility: {sample[3]} '
    elif sample[0] == 'd':
        text = f'title: {sample[1]} condition: {sample[2]} description: {sample[3]} '
    else:
        text = f'title: {sample[1]} condition: {sample[2]} eligibility: {sample[3]} description: {sample[4]} '
    return text


def get_all_judged(path_to_file, threshold):
    # path_to_file = '../../data/test_collection/qrels-clinical_trials.tsv'
    qrels = rf.read_qrel(path_to_file)
    out = {}
    for qid in qrels:
        out[qid] = {'pos': [], 'neg': []}
        for doc in qrels[qid]:
            if int(qrels[qid][doc]) > threshold:
                out[qid]['pos'].append(doc)
            else:
                out[qid]['neg'].append(doc)
    return out


def choose_document(qrels, trials, nctid2idx, ranklist):
    taget_num = min(len(qrels['pos']) * 8, 100)
    picked = set()
    tried = set()
    while len(picked) < taget_num and len(tried) < len(qrels['neg']):
        pick_num = len(qrels['pos']) - len(picked)
        neg_list = random.sample(qrels['neg'], pick_num)
        for docid in neg_list:
            if 'inclusion_list' in trials[nctid2idx[docid]] and 'desc' in trials[nctid2idx[docid]]:
                picked.add(docid)
            tried.add(docid)
    init_num = 300
    leftdoc = list(set(ranklist[:init_num]).difference(tried))
    final_pick_num = taget_num - len(picked)
    while len(leftdoc) < final_pick_num and init_num < 1000:
        init_num = min(init_num * 2, 1000)
        leftdoc = list(set(ranklist[:init_num]).difference(tried))
    if len(leftdoc) < final_pick_num:
        final_pick_num = len(leftdoc)
    picked = list(picked) + random.sample(leftdoc, final_pick_num)
    return (qrels['pos'], list(picked))


def get_model(model_path, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()
    tokenizer = T5BatchTokenizer(
        AutoTokenizer.from_pretrained(model_path, use_fast=False),
        batch_size=batch_size)
    reranker = MonoT5(model=model, tokenizer=tokenizer)
    return reranker


def run_monot5_on_judge(rankfile, path_to_file, path_to_pickle, query, outname):
    qrels = get_all_judged(path_to_file, 0)
    trials = pickle.load(open(path_to_pickle, 'rb'))
    nctid2idx = {i['number']: idx for idx, i in enumerate(trials)}

    outdir = './denseRetrieve/data/'
    ranklist = rf.read_resFile(rankfile)
    model_path = './crossEncoder/models/medMST5_ps_model'
    batch_size = 64
    reranker = get_model(model_path, batch_size)
    if os.path.exists(os.path.join(outdir, outname)):
        os.system('rm {}'.format(os.path.join(outdir, outname)))

    dtypeMap = {'i': 'inclusion_list', 'e': 'exclusion_list', 'd': 'desc'}

    with open(os.path.join(outdir, outname), 'a+') as outfile:
        qid_list = list(query.keys())
        for q_idx in tqdm(range(len(query))):
            idx = qid_list[q_idx]
            text = query[idx]
            text = re.sub(r'\r|\n|\t|\s\s+', ' ', text)
            query_class = Query(text)
            if idx not in qrels:
                continue
            docs = choose_document(qrels[idx], trials, nctid2idx, ranklist[idx])
            # run all pos, neg
            passages = [{'i': [], 'd': []}, {'i': [], 'd': []}]
            passages_doc = [{},{}]
            for i in range(2):
                texts = []
                for docid in docs[i]:
                    for dtype in ['inclusion_list', 'exclusion_list', 'desc']:
                        if dtype in trials[nctid2idx[docid]]:
                            for idxp, p in enumerate(trials[nctid2idx[docid]][dtype]):
                                if p:
                                    p = p.replace('..', '.')
                                    texts.append(Text(p, {'docid': docid + f'_{dtype[0]}_' + str(idxp)}, 0))
                reranked = reranker.rerank(query_class, texts)
                maxPassPerDoc = {}
                # maxp per doc
                for r in range(len(reranked)):
                    docid, dtype, idxp = reranked[r].metadata["docid"].split('_')
                    score = float(reranked[r].score)
                    if docid + dtype not in maxPassPerDoc:
                        maxPassPerDoc[docid + dtype] = (int(idxp), score)
                    elif maxPassPerDoc[docid + dtype][1] < score:
                        maxPassPerDoc[docid + dtype] = (int(idxp), score)

                for docid_dtype in maxPassPerDoc:
                    docid, dtype = docid_dtype[:-1], docid_dtype[-1]
                    ttype = dtypeMap[dtype]
                    title = trials[nctid2idx[docid]]['title'] if 'title' in trials[nctid2idx[docid]] else 'NA'
                    cond = trials[nctid2idx[docid]]['condition'] if 'condition' in trials[nctid2idx[docid]] else 'NA'
                    criteria = trials[nctid2idx[docid]][ttype][maxPassPerDoc[docid_dtype][0]]
                    title = re.sub(r'\r|\n|\t|\s\s+', ' ', str(title))
                    cond = re.sub(r'\r|\n|\t|\s\s+', ' ', str(cond))
                    criteria = re.sub(r'\r|\n|\t|\s\s+', ' ', str(criteria))
                    criteria.replace('..','.')
                    tmp_text = to_t5_input((dtype, title, cond, criteria))
                    dtype = 'i' if dtype == 'e' else dtype
                    passages[i][dtype].append(tmp_text)

            train_instance = {
                'query': text,
                'positives': passages[0]['i'] + passages[0]['d'],
                'negatives': passages[1]['i'] + passages[1]['d']
            }
            outfile.write(f'{json.dumps(train_instance)}\n')
            outfile.flush()

if __name__ == '__main__':
    random.seed(123)
    rankfile = './denseRetrieve/data/pyserini_tc_bm25.res'
    path_to_file = '../../data/test_collection/qrels-clinical_trials.tsv'
    path_to_pickle = './data/splits/clean_data_cfg_splits_63'
    path_to_query = '../../data/test_collection/topics-2014_2015-description.topics'
    query = rf.read_ts_topic(path_to_query)
    outname = 'tc_training.json'
    print(outname)
    run_monot5_on_judge(rankfile, path_to_file, path_to_pickle, query, outname)

    # path_to_file = '../../data/TRECCT2021/trec-ct2021-qrels.txt'
    # path_to_pickle = './data/splits/clean_data_cfg_splits_63_ct21'
    # path_to_query = '../../data/TRECCT2021/topics2021.xml'
    # outname = 'tripple_ct21.tsv'
    # query = rf.read_topics_ct21(path_to_query)
    # print(outname)
    # run_monot5_on_judge(path_to_file, path_to_pickle, query, outname)
