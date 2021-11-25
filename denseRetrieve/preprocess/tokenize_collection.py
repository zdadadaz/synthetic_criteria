import os
import json
import argparse
from tqdm import tqdm
from transformers import RobertaTokenizer
import pickle
import re
import numpy as np
from multiprocessing import Pool


def to_t5_input(sample):
    if sample[0] == 'i' or sample[0] == 'e':
        text = f'title: {sample[1]} condition: {sample[2]} eligibility: {sample[3]} '
    elif sample[0] == 'd':
        text = f'title: {sample[1]} condition: {sample[2]} description: {sample[3]} '
    else:
        text = f'title: {sample[1]} condition: {sample[2]} eligibility: {sample[3]} description: {sample[4]} '
    return text


def tokenize_one(tokenizer, seq_id, text, outFile):
    tokens = tokenizer.tokenize(text)
    tokenized = tokenizer.convert_tokens_to_ids(tokens)
    # encode(tokens, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
    outFile.write(json.dumps({"text_id": seq_id, "text": tokenized}))
    outFile.write("\n")


def tokenize_file(tokenizer, trials, output_file):
    total_size = len(trials)
    cnt = 0
    ids = []
    with open(output_file, 'w') as outFile:
        for idx, trial in tqdm(enumerate(trials), total=total_size,
                               desc=f"Tokenize"):
            docid = trial['number']
            title = trial['title'] if 'title' in trial else 'NA'
            cond = trial['condition'] if 'condition' in trial else 'NA'
            flag = True
            for etype in ['eligibility', 'desc']:
                if etype in trial and trial:
                    for idxp, p in enumerate(trial[etype]):
                        if p:
                            textT5 = to_t5_input((etype[0], title, cond, p))
                            textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
                            tokenize_one(tokenizer, cnt, textT5, outFile)
                            ids.append(docid + f'_{etype[0]}_' + f'{idxp:03}')
                            cnt += 1
                            flag = False
            if flag:  # no eligibility or description
                textT5 = to_t5_input(('e', title, cond, 'NA'))
                textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
                tokenize_one(tokenizer, cnt, textT5, outFile)
                ids.append(docid + f'_{etype[0]}_' + f'{-1:03}')
                cnt += 1
            outFile.flush()
    with open('./denseRetrieve/data/ct21/ance_token/nctid2id.json', 'w') as f:
        f.write('\n'.join(ids))

def tokenize_test(trial):
    out = []
    for idx, i in enumerate(trial['inclusion_list']):
        a = tokenizer.tokenize(i)
        encoded = {
            'text_id': trial['number'] + '_' + str(idx),
            'text': a
        }
        out.append(json.dumps(encoded))
    return '\n'.join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", type=str, default="./data/splits/clean_data_cfg_splits_42_ct21")
    parser.add_argument("--output_dir", type=str, default="./denseRetrieve/data/ct21/ance_token/corpus")
    parser.add_argument('--n_splits', type=int, default=10)
    args = parser.parse_args()

    path_to_pickle = args.pickle_file
    trials = pickle.load(open(path_to_pickle, 'rb'))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = RobertaTokenizer.from_pretrained('castorini/ance-msmarco-passage')
    tokenizer.do_lower_case = True

    n_lines = len(trials)
    if n_lines % args.n_splits == 0:
        split_size = int(n_lines / args.n_splits)
    else:
        split_size = int(n_lines / args.n_splits) + 1

    with Pool() as p:
        for i in range(args.n_splits):
            with open(os.path.join(args.output_dir, f'split{i:02d}.json'), 'w') as f:
                pbar = tqdm(trials[i * split_size: (i + 1) * split_size])
                pbar.set_description(f'split - {i:02d}')
                for jitem in p.imap(tokenize_test, pbar, chunksize=500):
                    f.write(jitem + '\n')
    # output_file = f"{args.output_dir}/collection.json"
    # tokenize_file(tokenizer, trials, output_file)
