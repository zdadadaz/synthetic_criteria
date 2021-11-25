import json
import random

import pandas as pd
from pygaggle.rerank.transformer import MonoT5, T5BatchTokenizer
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import torch
from nltk.tokenize import sent_tokenize
from pygaggle.rerank.base import Query, Text
import re
import os

class split_note:
    def __init__(self, dtype):
        self.dtype = dtype
        self.path_out = f'output/mimic/tripple_mimin_{self.dtype}_split.tsv'
        self.run_all()

    def run_all(self):
        df, out = self.read_file()
        model_path = './crossEncoder/models/t5base/medMST5_ps_model'
        batch_size = 64
        self.reranker = self.get_model(model_path, batch_size)
        self.split_all(df, out)

    def split_sentence(self, sentences, out):
        max_lenght = 6
        cur = []
        for l in sentences:
            if not l:
                continue
            if len(cur) < max_lenght:
                cur.append(l)
            else:
                out.append(' '.join(cur))
                for _ in range(max_lenght // 2):
                    cur.pop(0)
        if len(cur) <= max_lenght or not out or (out and out[-1] != ' '.join(cur)):
            out.append(' '.join(cur))

    def read_file(self):
        path_to_json = './output/mimic/mimic_dia.json'
        with open(path_to_json, 'r') as f:
            contents = json.loads(f.read())
        for type in ['train', 'test', 'val']:
            path_to_file = f'../ehr_section_prediction/mimicIII/DIA_PLUS_{type}.csv'
            if type == 'train':
                df = pd.read_csv(path_to_file)
            else:
                df.append(pd.read_csv(path_to_file))
        out = {}
        for j in contents:
            out[j['id']] = j
        return df, out

    def split_all(self, df, diseases):
        dtypes = [['pos', 'neg'], ['ret_pos', 'ret_neg']]
        dtype = dtypes[0] if self.dtype == 'temp' else dtypes[1]
        with open(self.path_out, 'w') as f:
            for i in tqdm(range(len(df))):
                qid, text = str(df.iloc[i, 0]), df.iloc[i, 1]
                if qid not in diseases:
                    continue
                text = text.lower()
                text = re.sub(r'\[([^\]]+)]', 'NA', text)
                text = re.sub(r'\r|\n|\t|\s\s+', ' ', text)
                disease = diseases[qid]
                sentences = sent_tokenize(text)
                passages = []
                self.split_sentence(sentences, passages)
                texts = []
                for idx, p in enumerate(passages):
                    texts.append(Text(p, {'docid': idx}, 0))
                # find the highest similarity score between criteria and note passage
                for idx, d in enumerate(disease[dtype[0]]):
                    query_class = Query(d)
                    reranked = self.reranker.rerank(query_class, texts)
                    maxidx, maxScore = -1, -10000
                    for r in range(len(reranked)):
                        docid = reranked[r].metadata["docid"]
                        score = float(reranked[r].score)
                        if maxidx == -1:
                            maxScore = score
                            maxidx = 0
                        elif maxScore < score:
                            maxScore = score
                            maxidx = int(docid)
                    pos_text = re.sub(r'\r|\n|\t|\s\s+', ' ', d)
                    neg_text = re.sub(r'\r|\n|\t|\s\s+', ' ', random.sample(disease[dtype[1]], 1)[0])
                    f.write("{}\t{}\t{}\n".format(passages[maxidx], pos_text, neg_text))
                f.flush()

    def get_model(self, model_path, batch_size):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()
        tokenizer = T5BatchTokenizer(
            AutoTokenizer.from_pretrained(model_path, use_fast=False),
            batch_size=batch_size)
        reranker = MonoT5(model=model, tokenizer=tokenizer)
        return reranker

class split_note_i2b2(split_note):
    def __init__(self, dtype, year):
        self.dtype = dtype
        self.year = year
        self.path_out = f'output/i2b2/tripple_i2b2_{self.year}_{self.dtype}_split.tsv'
        self.run_all()

    def read_file(self):
        path_to_json = f'./output/i2b2/{self.year}_dia.json'
        with open(path_to_json, 'r') as f:
            contents = json.loads(f.read())
        out = {}
        for j in contents:
            out[j['id']] = j

        path_to_file = f'../ehr_section_prediction/i2b2/synth/{self.year}_parsed.csv'
        df = pd.read_csv(path_to_file)
        df = df.dropna(how='any', subset=['text'], axis=0)
        df = df.drop(df[(df['header'] == 'Diagnosis')].index)
        df = df.groupby(['id'])['text'].apply(" ".join).reset_index()
        return df, out

def combine_all_file(dtype):
    path_output = f'./data/tripple/tripple_psu_{dtype}_split.tsv'
    path_to_dir = 'output'
    os.system('rm {}'.format(path_output))
    filelist = {}
    for path, subdirs, files in os.walk(path_to_dir):
        for name in files:
            if name.split('_')[-2] == dtype and name.split('_')[-1] == 'split.tsv':
                filelist[name.split('.')[0]] = os.path.join(path, name)
                os.system('cat {} >> {}'.format(os.path.join(path, name), path_output))


if __name__ == '__main__':
    # create tripple
    for dtype in ['temp', 'ret']:
        split_note(dtype)
        for i in ['2006', '2008', '2009', '2010', '2011', '2012', '2014']:
            split_note_i2b2(dtype, i)
        combine_all_file(dtype)

