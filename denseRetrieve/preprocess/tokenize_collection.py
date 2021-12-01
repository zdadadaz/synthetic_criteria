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

def tokenize_one_trial(trial):
    out = []
    docid = trial['number']
    title = trial['title'] if 'title' in trial else 'NA'
    cond = trial['condition'] if 'condition' in trial else 'NA'
    flag = True
    map2idx = {'e': 0, 'd': 1}
    print(docid)
    for etype in ['eligibility', 'desc']:
        if etype in trial and trial:
            for idxp, p in enumerate(trial[etype]):
                if p:
                    textT5 = to_t5_input((etype[0], title, cond, p))
                    textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
                    tokens = tokenizer.tokenize(textT5)
                    tokenized = tokenizer.convert_tokens_to_ids(tokens)
                    encoded = {
                        'text_id': docid[3:] + f'{map2idx[etype[0]]}' + f"{idxp:03}",
                        'text': tokenized
                    }
                    out.append(json.dumps(encoded))
                    flag = False
    if flag:  # no eligibility or description
        textT5 = to_t5_input(('e', title, cond, 'NA'))
        textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
        tokens = tokenizer.tokenize(textT5)
        tokenized = tokenizer.convert_tokens_to_ids(tokens)
        encoded = {
            'text_id': docid[3:] + f'{map2idx[etype[0]]}' + f"{999:03}",
            'text': tokenized
        }
        out.append(json.dumps(encoded))
    print(docid, len(out))
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
                for jitem in p.imap(tokenize_one_trial, pbar, chunksize=500):
                    f.write(jitem + '\n')
