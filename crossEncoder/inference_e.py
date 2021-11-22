from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from pygaggle.model import T5BatchTokenizer
import sys
import json

if 'src/' not in sys.path:
    sys.path.append('src/')
from utils import readfile as rf
from utils.eval import eval_set_args
import pickle
import os
import re
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import argparse
import torch

def eval(qrelsFile, res_path, out_method_name, out_path):
    eval = eval_set_args(qrelsFile, res_path)

    cmd = '../trec_eval-9.0.7/trec_eval -q {} {} > {}'.format(qrelsFile, res_path,
                                                              os.path.join(out_path, out_method_name) + '.qeval')
    os.system(cmd)

    out_txt = 'name,' + ','.join([i[0] for i in eval]) + '\n'
    out_txt += out_method_name + ',' + ','.join([str(i[1]) for i in eval]) + '\n'
    with open(os.path.join(out_path, f'{out_method_name}.eval'), 'w') as f:
        f.writelines(out_txt)


def to_t5_input(sample):
    if sample[4] == 'e':
        text = f'Query: {sample[0]} Document: title: {sample[1]} condition: {sample[2]} eligibility: {sample[3]} Relevant:'
    elif sample[4] == 'd':
        text = f'Query: {sample[0]} Document: title: {sample[1]} condition: {sample[2]} description: {sample[3]} Relevant:'
    elif sample[4] == 'ed':
        text = f'Query: {sample[0]} Document: title: {sample[1]} condition: {sample[2]} eligibility: {sample[3]} description: {sample[5]} Relevant:'
    return text


def write_result(qid, input_list, output_path, outname):
    trec = []
    cnt = 1
    for docid, score in input_list:
        trec.append("{}\t0\t{}\t{}\t{}\t{}\n".format(qid, docid, cnt, score, outname))
        cnt += 1

    with open(os.path.join(output_path, outname) + '.res', "a+") as f:
        f.writelines(trec)


def write_all_scores(qid, scores, output_path, outname, dtype):
    with open(os.path.join(output_path, outname) + f'_{dtype}_pscore.log', "a+") as f:
        for score, docid in scores:
            f.write("{}\t{}\t{}\n".format(qid, score, docid))
        f.flush()


def create_res_from_log(path_to_file, outdir, outname, outdir_eval):
    picked = set()
    out_scores = []
    for l in open(path_to_file, 'r'):
        if len(out_scores) >= 1000:
            break
        qid, score, docid_type = l.split('')
        docid_sub, dtype, pidx = docid_type.split('_')
        if docid_sub not in picked:
            picked.add(docid_sub)
            out_scores.append((docid_sub, score))
    write_result(qid, out_scores, outdir, outname)
    resfile = os.path.join(outdir, outname)
    eval(path_to_file, resfile, outname, outdir_eval)


def argfunc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='castorini/monot5-base-msmarco', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--outname", default=None, type=str, required=True,
                        help="medt5")
    parser.add_argument("--field", default=None, type=str, required=True,
                        help="e or ed")
    parser.add_argument("--batchsize", type=int, default=64)

    return parser.parse_args()

def get_model(model_path, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()
    tokenizer = T5BatchTokenizer(
        AutoTokenizer.from_pretrained(model_path, use_fast=False),
        batch_size=batch_size)
    reranker = MonoT5(model=model, tokenizer=tokenizer)
    return reranker


def inference():
    path_to_file = '../../data/TRECCT2021/trec_2021_binarized_qrels.txt'
    ############## choose the right split
    path_to_pickle = './data/splits/clean_data_cfg_splits_63_ct21'
    path_to_query = '../../data/TRECCT2021/topics2021.xml'
    path_to_run = './crossEncoder/runs/ielab-r2.res'
    outdir = './crossEncoder/runs'
    outdir_eval = './crossEncoder/eval'

    arg = argfunc()
    model_path = arg.base_model
    field_type = arg.field
    outname = arg.outname + '_' + field_type

    batch_size = int(arg.batchsize)
    print(model_path, field_type, outname)
    # outname = 'medt5'
    # field_type = 'e'  # e, ed
    # model_path = './crossEncoder/medMST5_ps_model'
    # batch_size=64

    resfile = os.path.join(outdir, outname) + '.res'
    if os.path.exists(resfile):
        os.system('rm {}'.format(resfile))

    logfile = os.path.join(outdir, outname) + '_individual_pscore.log'
    if os.path.exists(logfile):
        os.system('rm {}'.format(logfile))

    if not os.path.exists(outdir_eval):
        os.mkdir(outdir_eval)

    # # initialize
    query = rf.read_topics_ct21(path_to_query)
    res = rf.read_resFile(path_to_run)
    trials = pickle.load(open(path_to_pickle, 'rb'))
    nctid2idx = {i['number']: idx for idx, i in enumerate(trials)}
    reranker = get_model(model_path, batch_size)

    # prepare data
    qid_list = list(query.keys())
    for q_idx in tqdm(range(len(query))):
        qid = qid_list[q_idx]
        query_text = query[qid]
        query_text = re.sub(r'\r|\n|\t|\s\s+', ' ', query_text)
        query_class = Query(query_text)
        ids = []
        texts = []
        for docid in res[qid]:
            title = trials[nctid2idx[docid]]['title'] if 'title' in trials[nctid2idx[docid]] else 'NA'
            cond = trials[nctid2idx[docid]]['condition'] if 'condition' in trials[nctid2idx[docid]] else 'NA'
            flag = True
            for etype in ['eligibility', 'desc']:  # need add exlcusion
                if etype in trials[nctid2idx[docid]] and trials[nctid2idx[docid]][etype]:
                    for idxp, p in enumerate(trials[nctid2idx[docid]][etype]):
                        textT5 = to_t5_input((query_text, title, cond, p, 'e'))
                        textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
                        ids.append(docid + f'_{etype[0]}_' + str(idxp))
                        texts.append(Text(textT5, {'docid': docid + f'_{etype[0]}_' + str(idxp)}, 0))
                        flag = False
            if flag: # no eligibility or description
                textT5 = to_t5_input((query_text, title, cond, 'NA', 'e'))
                textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
                ids.append(docid + f'_{etype[0]}_' + str(idxp))
                texts.append(Text(textT5, {'docid': docid + f'_{etype[0]}_' + str(idxp)}, 0))

        reranked = reranker.rerank(query_class, texts)
        scores = []
        for r in range(len(reranked)):
            score = float(reranked[r].score)
            scores.append(score)
        scores = sorted(zip(scores, ids), key=lambda k: k[0], reverse=True)
        write_all_scores(qid, scores, outdir, outname, 'individual')

        picked = set()
        out_scores = []
        for score, doc_type in scores:
            if len(out_scores) >= 1000:
                break
            if doc_type == 'd':
                continue
            docid_sub, ftype, dixp = doc_type.split('_')
            if docid_sub not in picked:
                picked.add(docid_sub)
                out_scores.append((docid_sub, score))
        write_result(qid, out_scores, outdir, outname)
    eval(path_to_file, resfile, outname, outdir_eval)

if __name__ == '__main__':
    inference()