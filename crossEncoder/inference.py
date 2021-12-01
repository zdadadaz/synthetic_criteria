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
    if sample[3] == 'e':
        text = f'title: {sample[0]} condition: {sample[1]} eligibility: {sample[2]} Relevant:'
    elif sample[3] == 'd':
        text = f'title: {sample[0]} condition: {sample[1]} description: {sample[2]} Relevant:'
    elif sample[3] == 'ed':
        text = f'title: {sample[0]} condition: {sample[1]} eligibility: {sample[2]} description: {sample[4]} Relevant:'
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
        for r in range(len(scores)):
            docid_dtype = scores[r].metadata["docid"]
            score = float(scores[r].score)
            f.write("{}\t{}\t{}\n".format(qid, score, docid_dtype))
        f.flush()


def read_log(path_to_file):
    out_scores = {}
    for l in open(path_to_file, 'r'):
        qid, score, docid_type = l.split('\t')
        docid_sub, dtype, pidx = docid_type.split('_')
        if qid not in out_scores:
            out_scores[qid] = {}
        if docid_sub not in out_scores[qid]:
            out_scores[qid][docid_sub] = {'e':None, 'd':None}
        if not out_scores[qid][docid_sub][dtype]:
            out_scores[qid][docid_sub][dtype] = (int(pidx), float(score))
        elif out_scores[qid][docid_sub][dtype][1] < float(score):
            out_scores[qid][docid_sub][dtype] = (int(pidx), float(score))
    return out_scores

def argfunc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='t5-base', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--outname", default=None, type=str, required=True,
                        help="medt5")
    parser.add_argument("--log_path", default=None, type=str, required=True,
                        help="log path")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--model_parallel", type=int, default=0)

    return parser.parse_args()

def get_model(model_path, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()
    tokenizer = T5BatchTokenizer(
        AutoTokenizer.from_pretrained(model_path, use_fast=False),
        batch_size=batch_size, max_length=1024)
    reranker = MonoT5(model=model, tokenizer=tokenizer)

    return reranker

def inference():
    path_to_file = '../../data/TRECCT2021/trec_2021_binarized_qrels.txt'
    ############## choose the right split
    path_to_pickle = './data/splits/clean_data_cfg_splits_63_ct21'
    path_to_query = '../../data/TRECCT2021/topics2021.xml'
    path_to_run = './crossEncoder/data/ielab-r2.res'
    outdir = './crossEncoder/runs'
    outdir_eval = './crossEncoder/eval'

    arg = argfunc()
    model_path = arg.base_model
    field_type = 'ed'
    outname = arg.outname + '_' + field_type

    # outname = 'medt5'
    # field_type = 'ed'  # e, ed
    # model_path = './crossEncoder/medMST5_ps_model'
    batch_size = int(arg.batchsize)

    resfile = os.path.join(outdir, outname) + '.res'
    if os.path.exists(resfile):
        os.system('rm {}'.format(resfile))

    logfile = os.path.join(outdir, outname) + '_combine_pscore.log'
    if os.path.exists(logfile):
        os.system('rm {}'.format(logfile))

    if not os.path.exists(outdir_eval):
        os.mkdir(outdir_eval)

    # initialize
    query = rf.read_topics_ct21(path_to_query)
    trials = pickle.load(open(path_to_pickle, 'rb'))
    nctid2idx = {i['number']: idx for idx, i in enumerate(trials)}
    reranker = get_model(model_path, batch_size)

    # model parallel
    if arg.model_parallel != 0:
        reranker.model.parallelize()

    path_to_log = arg.log_path
    logs = read_log(path_to_log)

    dtype2trialtype = {'e': 'eligibility', 'd': 'desc'}

    # prepare data
    qid_list = list(query.keys())
    for q_idx in tqdm(range(len(query))):
        qid = qid_list[q_idx]
        query_text = query[qid]
        query_text = re.sub(r'\r|\n|\t|\s\s+', ' ', query_text)
        query_class = Query(query_text)
        ids = []
        texts = []
        log = logs[qid]
        for docid in log:
            title = trials[nctid2idx[docid]]['title'] if 'title' in trials[nctid2idx[docid]] else 'NA'
            cond = trials[nctid2idx[docid]]['condition'] if 'condition' in trials[nctid2idx[docid]] else 'NA'
            passages = []
            for field_type in log[docid]:
                if log[docid][field_type]:
                    p = 'NA' if log[docid][field_type][0] == -1 else trials[nctid2idx[docid]][dtype2trialtype[field_type]][log[docid][field_type][0]]
                    passages.append(p)
                else:
                    passages.append('NA')
            textT5 = to_t5_input((title, cond, passages[0], 'ed', passages[1]))
            textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
            ids.append(docid + '_ed')
            texts.append(Text(textT5, {'docid': docid + '_ed'}, 0))

        reranked = reranker.rerank(query_class, texts)
        write_all_scores(qid, reranked, outdir, outname, 'combine')

        picked = set()
        out_scores = []
        for r in range(len(reranked)):
            doc_type = reranked[r].metadata["docid"]
            score = float(reranked[r].score)
            if len(out_scores) >= 1000:
                break
            docid_sub, ftype = doc_type.split('_')
            if docid_sub not in picked:
                picked.add(docid_sub)
                out_scores.append((docid_sub, score))
        write_result(qid, out_scores, outdir, outname)
    eval(path_to_file, resfile, outname, outdir_eval)


if __name__ == '__main__':
    inference()
