import copy
import pickle
import os
import json
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

def split_sentence(sentences, out):
    max_lenght = 6
    cur = []
    for l in sentences:
        if not l:
            continue
        if len(cur) < max_lenght:
            cur.append(l)
        else:
            out.append(' '.join(cur))
            for _ in range(max_lenght//2):
                cur.pop(0)
    if len(cur) <= max_lenght or not out or (out and out[-1] != ' '.join(cur)):
        out.append(' '.join(cur))

def gen_eligibility(trial):
    out = []
    alltexts = []
    for dtype in ['inclusion_list', 'exclusion_list']:
        if dtype in trial and trial[dtype] and trial[dtype][0]:
            alltexts += trial[dtype]
    alltexts = '. '.join(alltexts)
    allsentence = sent_tokenize(alltexts)
    split_sentence(allsentence, out)
    return out

def gen_eligibility_ie(trial):
    out = []
    for dtype in ['inclusion_list', 'exclusion_list']:
        sub_out = []
        alltexts = []
        if dtype in trial and trial[dtype] and trial[dtype][0]:
            alltexts += trial[dtype]
        if alltexts:
            alltexts = '. '.join(alltexts)
            allsentence = sent_tokenize(alltexts)
            split_sentence(allsentence, sub_out)
            sub_out = [dtype[:-5] + ': ' + i for i in sub_out]
            out += sub_out
    return out

def split_document(path_to_dir, path_to_pickle, path_out_pickle):
    trials = pickle.load(open(path_to_pickle, 'rb'))
    filelist = {}
    for path, subdirs, files in os.walk(path_to_dir):
        for name in files:
            if name.split('.')[-1] == 'json':
                filelist[name.split('.')[0]] = os.path.join(path, name)

    for idx in tqdm(range(len(trials))):
        trial = trials[idx]
        with open(filelist[trial['number']], 'r') as j:
            jfile = json.loads(j.read())
        title = jfile['ot'] if 'ot' in jfile else jfile['bt']
        trial['title'] = title
        desc = jfile['dd'] if 'dd' in jfile else ''
        desc = jfile['bs'] if not desc and 'bs' in desc else desc
        out = []
        sentences = sent_tokenize(desc)
        split_sentence(sentences, out)
        trial['desc'] = out
        # out = gen_eligibility(trial)
        out = gen_eligibility_ie(trial)
        trial['eligibility'] = out
    pickle.dump(trials, open(path_out_pickle, 'wb'))


if __name__ == '__main__':
    path_to_dir = '../../data/test_collection/clinicaltrials_json_cond_sym_flat'
    path_to_pickle = '../ctCriteria/utils_parse_cfg/parsed_data/clean_data_cfg'
    path_out_pickle = 'data/splits/clean_data_cfg_splits_63_ie'
    split_document(path_to_dir, path_to_pickle, path_out_pickle)

    path_to_dir = '../../data/TRECCT2021/clinicaltrials_json_cond'
    path_to_pickle = '../ctCriteria/utils_parse_cfg/parsed_data/clean_data_cfg_ct21'
    path_out_pickle = 'data/splits/clean_data_cfg_splits_63_ct21_ie'
    split_document(path_to_dir, path_to_pickle, path_out_pickle)
