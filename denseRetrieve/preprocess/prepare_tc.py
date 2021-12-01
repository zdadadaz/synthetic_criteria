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
import heapq


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
    taget_num = min(len(qrels['pos']) * 10, 100)
    picked = set()
    tried = set()
    # choose from judged negative
    while len(picked) < taget_num and len(tried) < len(qrels['neg']):
        pick_num = min(taget_num - len(picked), len(list(set(qrels['neg']).difference(tried))))
        neg_list = random.sample(list(set(qrels['neg']).difference(tried)), pick_num)
        for docid in neg_list:
            if 'eligibility' in trials[nctid2idx[docid]] and 'desc' in trials[nctid2idx[docid]] and \
                    trials[nctid2idx[docid]]['eligibility'][0] and trials[nctid2idx[docid]]['desc'][0]:
                picked.add(docid)
            tried.add(docid)
    if len(picked) == taget_num:
        return (qrels['pos'], list(picked))

    # choose before 300
    init_num = 300
    leftdoc = list(set(ranklist[:init_num]).difference(tried))
    for docid in leftdoc:
        if len(picked) >= taget_num:
            return (qrels['pos'], list(picked))
        if 'eligibility' in trials[nctid2idx[docid]] and 'desc' in trials[nctid2idx[docid]] and \
                trials[nctid2idx[docid]]['eligibility'][0] and trials[nctid2idx[docid]]['desc'][0]:
            picked.add(docid)
        tried.add(docid)

    # last random choose
    if len(picked) < taget_num:
        final_pick_num = taget_num - len(picked)
        picked = list(picked) + random.sample(list(set(ranklist[init_num:(init_num * 2)]).difference(tried)),
                                              final_pick_num)
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
    model_path = './crossEncoder/models/t5base/medMST5_ps_model'
    batch_size = 64
    reranker = get_model(model_path, batch_size)
    if os.path.exists(os.path.join(outdir, outname)):
        os.system('rm {}'.format(os.path.join(outdir, outname)))

    dtypeMap = {'e': 'eligibility', 'd': 'desc'}

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
            passages = [{'e': [], 'd': []}, {'e': [], 'd': []}]
            doc2ed = [{}, {}]
            for i in range(2):
                texts = []
                for docid in docs[i]:
                    for dtype in ['eligibility', 'desc']:
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
                    criteria.replace('..', '.')
                    tmp_text = to_t5_input((dtype, title, cond, criteria))
                    passages[i][dtype].append(tmp_text)
                    if docid not in doc2ed[i]:
                        doc2ed[i][docid] = {'e': None, 'd': None}
                    doc2ed[i][docid][dtype] = (title, cond, criteria)

            for dtype in ['e', 'd']:
                train_instance = {
                    'query': text,
                    'positives': passages[0][dtype],
                    'negatives': passages[1][dtype]
                }
                flag = True
                for ttype in train_instance:
                    if not train_instance[ttype]:
                        flag = False
                if flag:
                    outfile.write(f'{json.dumps(train_instance)}\n')
            train_instance = {
                'query': text,
                'positives': [],
                'negatives': []
            }
            for i in range(2):
                dtype = 'positives' if i == 0 else 'negatives'
                for docid in doc2ed[i]:
                    if doc2ed[i][docid]['e'] and doc2ed[i][docid]['d']:
                        tmp_text = to_t5_input(('ed', doc2ed[i][docid]['e'][0], doc2ed[i][docid]['e'][1],
                                                doc2ed[i][docid]['e'][2], doc2ed[i][docid]['d'][2]))
                        train_instance[dtype].append(tmp_text)
            flag = True
            for ttype in train_instance:
                if not train_instance[ttype]:
                    flag = False
            if flag:
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

    path_to_file = '../../data/TRECCT2021/trec-ct2021-qrels.txt'
    path_to_pickle = './data/splits/clean_data_cfg_splits_63_ct21'
    path_to_query = '../../data/TRECCT2021/topics2021.xml'
    outname = 'tripple_ct21.tsv'
    query = rf.read_topics_ct21(path_to_query)
    print(outname)
    run_monot5_on_judge(path_to_file, path_to_pickle, query, outname)
