import sys

if 'src/' not in sys.path:
    sys.path.append('src/')
from utils import readfile as rf
import pickle
import argparse
from tqdm import tqdm
import re

parser = argparse.ArgumentParser()
parser.add_argument("--queries", type=str, required=True,
                    help="tsv file with two columns, <query_id> and <query_text>")
parser.add_argument("--t5_output", type=str, required=True,
                    help="tsv file with two columns, <label> and <score>")
parser.add_argument("--t5_output_ids", type=str, required=True,
                    help="tsv file with two columns, <query_id> and <doc_id>")
parser.add_argument("--path_to_pickle", type=str, required=True)
parser.add_argument("--t5_input", type=str, required=True,
                    help="path to store t5_input, txt format")
parser.add_argument("--t5_input_ids", type=str, required=True,
                    help="path to store the query-doc ids of t5_input, tsv format")
args = parser.parse_args()


def to_t5_input(sample):
    if sample[3] == 'e':
        text = f'title: {sample[0]} condition: {sample[1]} eligibility: {sample[2]}'
    elif sample[3] == 'd':
        text = f'title: {sample[0]} condition: {sample[1]} description: {sample[2]}'
    elif sample[3] == 'ed':
        text = f'title: {sample[0]} condition: {sample[1]} eligibility: {sample[2]} description: {sample[4]}'
    return text


def read_log(path_to_t5_output, path_t5_output_ids):
    out_scores = {}
    with open(args.t5_output_ids) as f_gt, open(args.t5_output) as f_pred:
        for line_gt, line_pred in zip(f_gt, f_pred):
            qid, docid_type = line_gt.strip().split('\t')
            docid_sub, dtype, pidx= docid_type.split('_')
            _, score = line_pred.strip().split('\t')
            if qid not in out_scores:
                out_scores[qid] = {}
            if docid_sub not in out_scores[qid]:
                out_scores[qid][docid_sub] = {'e': None, 'd': None}
            if not out_scores[qid][docid_sub][dtype]:
                out_scores[qid][docid_sub][dtype] = (int(pidx), float(score))
            elif out_scores[qid][docid_sub][dtype][1] < float(score):
                out_scores[qid][docid_sub][dtype] = (int(pidx), float(score))
    return out_scores


def main():
    path_to_query = args.queries
    path_to_run = args.run
    path_to_pickle = args.path_to_pickle
    # # initialize
    if '2021' in path_to_query:
        query = rf.read_topics_ct21(path_to_query)
    else:
        query = rf.read_ts_topic(path_to_query)
    trials = pickle.load(open(path_to_pickle, 'rb'))
    nctid2idx = {i['number']: idx for idx, i in enumerate(trials)}

    logs = read_log(args.scores)
    dtype2trialtype = {'e': 'eligibility', 'd': 'desc'}

    print("Writing t5 input and ids")
    # prepare data
    # prepare data
    qid_list = list(query.keys())
    with open(args.t5_input, 'w') as fout_t5, open(args.t5_input_ids, 'w') as fout_tsv:
        for q_idx in tqdm(range(len(query))):
            qid = qid_list[q_idx]
            query_text = query[qid]
            query_text = re.sub(r'\r|\n|\t|\s\s+', ' ', query_text)
            log = logs[qid]
            for docid in log:
                title = trials[nctid2idx[docid]]['title'] if 'title' in trials[nctid2idx[docid]] else 'NA'
                cond = trials[nctid2idx[docid]]['condition'] if 'condition' in trials[nctid2idx[docid]] else 'NA'
                passages = []
                for field_type in log[docid]:
                    if log[docid][field_type]:
                        p = 'NA' if log[docid][field_type][0] == -1 else \
                            trials[nctid2idx[docid]][dtype2trialtype[field_type]][log[docid][field_type][0]]
                        passages.append(p)
                    else:
                        passages.append('NA')
                textT5 = to_t5_input((title, cond, passages[0], 'ed', passages[1]))
                textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
                fout_t5.write(f'Query: {query_text} Document: {textT5} Relevant:\n')
                fout_tsv.write(f'{qid}\t{docid}\n')

if __name__ == '__main__':
    main()