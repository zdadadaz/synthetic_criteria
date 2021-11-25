import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET
import re

def read_topics_ct21(path_to_topics) -> dict:
    '''
    return a dict that maps qid, content pair
    '''
    topics = defaultdict(dict)
    tree = ET.parse(path_to_topics)
    root = tree.getroot()
    for topic in root:
        idx = topic.attrib['number']
        topics[idx] = topic.text
    return topics

def read_res(path_to_file):
    out = {}
    for l in open(path_to_file):
        qid, docid, score = l.strip().split('\t')
        if qid not in out:
            out[qid] = []
        out[qid].append((docid, float(score)))
    return out

def write_res(out_dict, path_out):
    out = []
    for qid in out_dict:
        outlist = sorted(out_dict[qid], key=lambda k:k[1], reverse=True)
        exist = set()
        for idx, (docid, score) in enumerate(outlist):
            if docid not in exist:
                out.append(f'{qid}\tQ0\t{docid}\t{idx+1}\t{score}\tmethod\n')
                exist.add(docid)
    with open(path_out, 'w') as f:
        f.writelines(out)

def tevatron_res_to_trec():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=False,
                        help="path to trevton res file")
    args = parser.parse_args()
    path_to_file = args.file
    # path_to_file = 'denseRetrieve/data/bioMSmarco/ranking/ance_rank.txt'
    # out_path = 'denseRetrieve/data/bioMSmarco/ranking/ance_rank_trec.res'
    out_path = path_to_file[-3:] + 'res'
    out_dict = read_res(path_to_file)
    write_res(out_dict, out_path)

def modify_query_format():
    path_to_file = '../../data/TRECCT2021/topics2021.xml'
    query = read_topics_ct21(path_to_file)
    out_path = '../../data/TRECCT2021/topics2021.tsv'
    out = ['{}\t{}\n'.format(qid, re.sub(r'\r|\n|\t|\s\s+', ' ', text)) for qid, text in query.items()]
    with open(out_path, 'w') as f:
        f.writelines(out)

if __name__ == '__main__':
    tevatron_res_to_trec()
    # modify_query_format()