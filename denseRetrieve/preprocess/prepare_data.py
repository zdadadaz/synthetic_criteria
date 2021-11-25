import os
import json
import sys
if 'src/' not in sys.path:
    sys.path.append('src/')
from utils import readfile as rf
import random
import pandas as pd
import re

def read_i2b2_file(year):
    path_to_file = f'../ehr_section_prediction/i2b2/synth/{year}_parsed.csv'
    df = pd.read_csv(path_to_file, dtype={"id": str})
    df = df.dropna(how='any', subset=['text'], axis=0)
    df = df.drop(df[(df['header'] == 'Diagnosis')].index)
    df = df.groupby(['id'])['text'].apply(" ".join).reset_index()
    return df

def read_mimic_file():
    for type in ['train', 'test', 'val']:
        path_to_file = f'../ehr_section_prediction/mimicIII/DIA_PLUS_{type}.csv'
        if type == 'train':
            df = pd.read_csv(path_to_file, dtype={"id": str})
        else:
            df.append(pd.read_csv(path_to_file))
    return df

def create_json_psuCriteria_tevatron():
    path_to_dirs = ['./output/mimic/', './output/i2b2']
    path_out_list = './denseRetrieve/data/psuId_mapping.json'
    if os.path.exists('./denseRetrieve/data/psu_temp_training.json'):
        os.system('rm {}'.format('./denseRetrieve/data/psu_temp_training.json'))
    if os.path.exists('./denseRetrieve/data/psu_ret_training.json'):
        os.system('rm {}'.format('./denseRetrieve/data/psu_ret_training.json'))

    filelist = []
    for dir in path_to_dirs:
        for path, subdirs, files in os.walk(dir):
            for name in files:
                if 'dia' in name and name.split('_')[-1] == '8neg.json':
                    filelist.append(os.path.join(path, name))
    dfs = {}
    dfs['mimic'] = read_mimic_file()
    for i in ['2006', '2008', '2009', '2010', '2011', '2012', '2014']:
        dfs[i] = read_i2b2_file(i)

    fout = [open('./denseRetrieve/data/psu_temp_training.json', 'a+'), open('./denseRetrieve/data/psu_ret_training.json', 'a+')]
    idmap, cnt = {}, 1
    for file in filelist:
        with open(file, 'r') as f:
            contents = json.loads(f.read())
        for doc in contents:
            df_name = file.split('/')[-1].split('_')[0]
            org_id = file.split('/')[-1].split('_')[0] + '_' + doc['id']
            text = dfs[df_name][dfs[df_name]['id'] == doc['id']]['text'].iloc[0]
            text = text.lower()
            text = re.sub(r'\[([^\]]+)]', 'NA', text)
            text = re.sub(r'\r|\n|\t|\s\s+', ' ', text)
            if org_id not in idmap:
                idmap[org_id] = cnt
                cnt += 1
            for idx, dtype in enumerate([['pos','neg'],['ret_pos', 'ret_neg']]):
                if dtype[0] in doc and dtype[1] in doc:
                    train_instance = {
                        'query': text,
                        'positives': doc[dtype[0]],
                        'negatives': doc[dtype[1]]
                    }
                    flag = True
                    for ttype in train_instance:
                        if not train_instance[ttype]:
                            flag = False
                    if flag:
                        fout[idx].write(f'{json.dumps(train_instance)}\n')

    fout[0].close()
    fout[1].close()
    json_object = json.dumps(idmap,indent=2)
    with open(path_out_list, "w") as outfile:
        outfile.write(json_object)

def create_json_biomsmarco_tevatron():
    path_to_rank = './denseRetrieve/data/bioMSmarco/ranking/ance_rank_trec.res'
    path_to_qrels = '../../data/bio-MSmarco/qrels.train.tsv'
    path_to_query = '../../data/bio-MSmarco/queries.train.tsv'
    path_to_collection = '../../data/bio-MSmarco/collection.tsv'
    out_path = './denseRetrieve/data/bio_training.json'

    ranklist = rf.read_resFile(path_to_rank)
    qrels = rf.read_qrel(path_to_qrels)
    queries = {l.strip().split('\t')[0]:l.strip().split('\t')[1] for l in open(path_to_query)}
    collection = {l.strip().split('\t')[0]: l.strip().split('\t')[1] for l in open(path_to_collection)}

    if os.path.exists(out_path):
        os.system('rm {}'.format(out_path))

    with open(out_path, 'a+') as outfile:
        for qid in queries:
            text = queries[qid]
            pos_list = []
            doc_list = []
            for pos in qrels[qid]:
                pos_list.append(pos)
            negs = random.sample(list(set(ranklist[qid][:300]).difference(set(qrels[qid]))), min(100, 8 * len(qrels[qid])))
            for neg in negs:
                doc_list.append(collection[neg])
            train_instance = {
                'query': text,
                'positives': pos_list,
                'negatives': doc_list
            }
            outfile.write(f'{json.dumps(train_instance)}\n')
            outfile.flush()


if __name__ == '__main__':
    create_json_psuCriteria_tevatron()
    # create_json_biomsmarco_tevatron()