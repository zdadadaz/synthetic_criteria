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
parser.add_argument("--run", type=str, required=True,
                    help="tsv file with three columns <query_id>, <doc_id> and <rank>")
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

def main():
    path_to_query = args.queries
    path_to_run = args.run
    path_to_pickle = args.path_to_pickle
    # # initialize
    if '2021' in path_to_query:
        query = rf.read_topics_ct21(path_to_query)
    else:
        query = rf.read_ts_topic(path_to_query)
    res = rf.read_resFile(path_to_run)
    trials = pickle.load(open(path_to_pickle, 'rb'))
    nctid2idx = {i['number']: idx for idx, i in enumerate(trials)}

    print("Writing t5 input and ids")
    # prepare data
    qid_list = list(query.keys())
    with open(args.t5_input, 'w') as fout_t5, open(args.t5_input_ids, 'w') as fout_tsv:
        for q_idx in tqdm(range(len(query))):
            qid = qid_list[q_idx]
            query_text = query[qid]
            query_text = re.sub(r'\r|\n|\t|\s\s+', ' ', query_text)
            query_text = query_text.strip()
            texts = []
            # can remove after run ct judgment
            if qid not in res:
                continue
            for docid in res[qid]:
                title = trials[nctid2idx[docid]]['title'] if 'title' in trials[nctid2idx[docid]] else 'NA'
                cond = trials[nctid2idx[docid]]['condition'] if 'condition' in trials[nctid2idx[docid]] else 'NA'
                flag = True
                for etype in ['eligibility', 'desc']:
                    if etype in trials[nctid2idx[docid]] and trials[nctid2idx[docid]][etype]:
                        for idxp, p in enumerate(trials[nctid2idx[docid]][etype]):
                            if p:
                                textT5 = to_t5_input((title, cond, p, etype[0]))
                                textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
                                texts.append(textT5)
                                fout_t5.write(f'Query: {query_text} Document: {textT5} Relevant:\n')
                                docidout = docid + f'_{etype[0]}_' + str(idxp)
                                fout_tsv.write(f'{qid}\t{docidout}\n')
                                flag = False
                if flag:  # no eligibility or description
                    textT5 = to_t5_input((title, cond, 'NA', 'e'))
                    textT5 = re.sub(r'\r|\n|\t|\s\s+', ' ', str(textT5))
                    fout_t5.write(f'Query: {query_text} Document: {textT5} Relevant:\n')
                    docidout = docid + '_e_' + str(-1)
                    fout_tsv.write(f'{qid}\t{docidout}\n')

if __name__ == '__main__':
    main()
