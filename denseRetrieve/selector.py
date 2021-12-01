import argparse
import glob
import json

import numpy as np
import faiss
import os
import re
import warnings
import torch
from itertools import chain
from tqdm import tqdm
from utils import read_resFile

def argfunc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outname", default=None, type=str, required=True,
                        help="medt5")
    parser.add_argument("--query_reps", default=None, type=str, required=True,
                        help="query_reps path")
    parser.add_argument("--passage_reps", default=None, type=str, required=True,
                        help="passage_reps path")

    return parser.parse_args()

class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray):
        self.last_idx = 0
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index
        self.look_up = [{},{}]
        self.lookuplist = []

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def add(self, p_reps: np.ndarray, lookup):
        self.index.add(p_reps)
        self.lookuplist += lookup
        self.last_idx += 1

    def addlookup(self, docid, start_idx, dtype):
        if docid not in self.look_up[dtype]:
            self.look_up[dtype][docid] = [-1, -1]
        if self.look_up[dtype][docid][0] == -1:
            self.look_up[dtype][docid][0] = start_idx
        else:
            self.look_up[dtype][docid][1] = start_idx

    def get_doc_data(self, doc, dtype):
        if doc[3:] not in self.look_up[dtype]:
            return None
        st, en = self.look_up[dtype][doc[3:]]
        if en == -1:
            en = st
        return self.index.reconstruct_n(st, en-st+1), self.lookuplist[st:(en+1)]


def search_queries(retriever, q_reps, q_lookup, res):
    cosines = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    out = {}
    for idx, qid in enumerate(q_lookup):
        out[qid] = {}
        for doc in res[qid]:
            out[qid][doc] = [None, None]
            for dtype in range(2):
                if retriever.get_doc_data(doc, dtype):
                    doc_reps, doc_idx = retriever.get_doc_data(doc, dtype)
                    similarity = cosines(torch.tensor(q_reps[idx]), torch.tensor(doc_reps))
                    maxpass = similarity.float().numpy().argmax()
                    out[qid][doc][dtype] = doc_idx[maxpass]
    return out

def read_pt2index(args):
    index_files = glob.glob(args.passage_reps)
    print(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = torch.load(index_files[0])
    retriever = BaseFaissIPRetriever(p_reps_0.float().numpy())

    shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    for p_reps, p_lookup in shards:
        start = retriever.last_idx
        retriever.add(p_reps.float().numpy(), p_lookup)
        for idx, doc in enumerate(p_lookup):
            start_idx = start + idx
            docid, doctype, pid = doc[:8], doc[8], doc[9:]
            retriever.addlookup(docid, start_idx, int(doctype))
    return retriever

def rerank():
    path_to_res = 'crossEncoder/data/ielab-r2.res'
    args = argfunc()
    field_type = 'ed'
    outdir_selector = f'denseRetrieve/data/ct21/selector/{args.outname}'

    retriever = read_pt2index(args)

    q_reps, q_lookup = torch.load(args.query_reps)
    q_reps = q_reps.float().numpy()
    res = read_resFile(path_to_res)
    out = search_queries(retriever, q_reps, q_lookup, res)

    if not os.path.exists(outdir_selector):
        os.makedirs(outdir_selector, exist_ok=True)
    with open(outdir_selector + f'/{args.outname}.json', 'w') as f:
        f.write(json.dumps(out, indent=2))

if __name__ == '__main__':
    rerank()