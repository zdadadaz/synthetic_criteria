import os
import json
import sys
if 'src/' not in sys.path:
    sys.path.append('src/')
from utils import readfile as rf
import random

def create_json_psuCriteria_tevatron():
    path_to_dirs = ['./output/mimic/', './output/i2b2']
    path_out_list = './denseRetrieve/data/psuId_mapping.json'
    if os.path.exists('./denseRetrieve/data/psu_temp_training.json'):
        os.system('rm {}'.format('./denseRetrieve/data/psu_temp_training.json'))
    if os.path.exists('./denseRetrieve/data/psu_ret_training.json'):
        os.system('rm {}'.format('./denseRetrieve/data/psu_ret_training.json'))

    for dir in path_to_dirs:
        filelist = []
        for path, subdirs, files in os.walk(dir):
            for name in files:
                if 'dia' in name and name.split('.')[-1] == 'json':
                    filelist.append(os.path.join(path, name))

    fout = [open('./denseRetrieve/data/psu_temp_training.json', 'a+'), open('./denseRetrieve/data/psu_ret_training.json', 'a+')]
    idmap, cnt = {}, 1
    for file in filelist:
        with open(file, 'r') as f:
            contents = json.loads(f.read())
        for doc in contents:
            org_id = file.split('/')[-1].split('_')[0] + '_' + doc['id']
            if org_id not in idmap:
                idmap[org_id] = cnt
                cnt += 1
            for idx, dtype in enumerate([['pos','neg'],['ret_pos', 'ret_neg']]):
                if dtype[0] in doc and dtype[1] in doc:
                    train_instance = {
                        'query': cnt-1,
                        'positives': doc[dtype[0]],
                        'negatives': doc[dtype[1]]
                    }
                    fout[idx].write(f'{json.dumps(train_instance)}\n')

    fout[0].close()
    fout[1].close()
    json_object = json.dumps(idmap,indent=2)
    with open(path_out_list, "w") as outfile:
        outfile.write(json_object)

def create_json_biomsmarco_tevatron():
    path_to_rank = './denseRetrieve/data/bioMSmarco/ranking/ance_rank_trec.txt'
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
            negs = random.sample(list(set(ranklist[:300]).difference(set(qrels[qid]))), min(100, 8 * len(qrels[qid])))
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
    # create_json_psuCriteria_tevatron()
    create_json_biomsmarco_tevatron()