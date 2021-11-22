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


def choose_document(qrels, trials, nctid2idx):
    # return (qrels['pos'][:1], qrels['neg'][:1])
    if len(qrels['pos']) > len(qrels['neg']):
        return (qrels['pos'], qrels['neg'])
    else:
        picked = set()
        tried = set()
        while len(picked) < len(qrels['pos']) and len(tried) < len(qrels['neg']):
            pick_num = len(qrels['pos']) - len(picked)
            neg_list = random.sample(qrels['neg'], pick_num)
            for docid in neg_list:
                if 'inclusion_list' in trials[nctid2idx[docid]] and 'desc' in trials[nctid2idx[docid]]:
                    picked.add(docid)
                tried.add(docid)
        if len(picked) < len(qrels['pos']):
            picked = list(picked) + random.sample(list(tried.difference(picked)), len(qrels['pos']) - len(picked))
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


def run_monot5_on_judge(path_to_file, path_to_pickle, query, outname):
    qrels = get_all_judged(path_to_file, 0)
    trials = pickle.load(open(path_to_pickle, 'rb'))
    nctid2idx = {i['number']: idx for idx, i in enumerate(trials)}

    model_path = './crossEncoder/models/t5base/medMST5_ps_model'
    batch_size = 64
    reranker = get_model(model_path, batch_size)

    dtypeMap = {'e': 'eligibility', 'd': 'desc'}

    with open(os.path.join('./data/tripple', outname), 'w') as outfile:
        qid_list = list(query.keys())
        for q_idx in tqdm(range(len(query))):
            idx = qid_list[q_idx]
            text = query[idx]
            text = re.sub(r'\r|\n|\t|\s\s+', ' ', text)
            query_class = Query(text)
            if idx not in qrels:
                continue
            docs = choose_document(qrels[idx], trials, nctid2idx)
            # run all pos, neg
            passages = [{}, {}]
            for i in range(2):
                texts = []
                for docid in docs[i]:
                    for dtype in ['eligibility', 'desc']:
                        if dtype in trials[nctid2idx[docid]]:
                            for idxp, p in enumerate(trials[nctid2idx[docid]][dtype]):
                                if p:
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
                    criteria = criteria.replace('..','.')
                    if docid not in passages[i]:
                        passages[i][docid] = {}
                    passages[i][docid][dtype] = (title, cond, criteria)
            # write out tripple: query, title, condition, pos inclusion/description, neg inclusion/description
            neg_doc_list = list(passages[1].keys())
            for docid in passages[0]:
                # separate i and d
                for dtype in ['e', 'd']:
                    out_script = [text]
                    if dtype not in passages[0][docid]:
                        continue
                    # pos
                    for p in passages[0][docid][dtype]:
                        out_script.append(p)
                    # rand neg
                    negdoc = random.sample(neg_doc_list, 1)[0]
                    while dtype not in passages[1][negdoc]:
                        negdoc = random.sample(neg_doc_list, 1)[0]
                    for p in passages[1][negdoc][dtype]:
                        out_script.append(p)
                    outfile.write('\t'.join(out_script) + f'\t{dtype}' '\n')
                # combine i and d
                chosen_docid = docid
                for dtype in ['e']:
                    out_script = [text]
                    for pos_neg in range(2):
                        if pos_neg == 1:# neg
                            chosen_docid = random.sample(neg_doc_list, 1)[0]
                            while dtype not in passages[1][chosen_docid] or 'd' not in passages[pos_neg][chosen_docid]:
                                chosen_docid = random.sample(neg_doc_list, 1)[0]
                        if chosen_docid not in passages[pos_neg] or dtype not in passages[pos_neg][chosen_docid] or 'd' not in \
                                passages[pos_neg][chosen_docid]:
                            break
                        for p in passages[pos_neg][chosen_docid][dtype]:
                            out_script.append(p)
                        out_script.append(passages[pos_neg][chosen_docid]['d'][-1])
                    if len(out_script) >= 9:
                        outfile.write('\t'.join(out_script) + f'\t{dtype}d' + '\n')
                outfile.flush()


if __name__ == '__main__':
    random.seed(123)
    path_to_file = '../../data/test_collection/qrels-clinical_trials.tsv'
    path_to_pickle = './data/splits/clean_data_cfg_splits_42'
    path_to_query = '../../data/test_collection/topics-2014_2015-description.topics'
    query = rf.read_ts_topic(path_to_query)
    outname = 'tripple_tc.tsv'
    print(outname)
    run_monot5_on_judge(path_to_file, path_to_pickle, query, outname)

    # path_to_file = '../../data/TRECCT2021/trec-ct2021-qrels.txt'
    # path_to_pickle = './data/splits/clean_data_cfg_splits_42_ct21'
    # path_to_query = '../../data/TRECCT2021/topics2021.xml'
    # outname = 'tripple_ct21.tsv'
    # query = rf.read_topics_ct21(path_to_query)
    # print(outname)
    # run_monot5_on_judge(path_to_file, path_to_pickle, query, outname)
