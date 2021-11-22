import pyterrier as pt
import re
import pandas as pd
import json
import os
from eval import eval_set_args
import readfile as rf


def eval(qrelsFile, res_path, out_method_name, out_path):
    eval = eval_set_args(qrelsFile, res_path)

    cmd = '../trec_eval-9.0.7/trec_eval -q {} {} > {}'.format(qrelsFile, res_path, os.path.join(out_path,out_method_name) + '.qeval')
    os.system(cmd)

    out_txt = 'name,' + ','.join([i[0] for i in eval]) + '\n'
    out_txt += out_method_name + ',' + ','.join([str(i[1]) for i in eval]) + '\n'
    with open(os.path.join(out_path, f'{out_method_name}.eval'), 'w') as f:
        f.writelines(out_txt)


def write_res(res, out_path, outname):
    out = []
    for i in range(len(res)):
        out.append(
            "{}\tQ0\t{}\t{}\t{}\truns\n".format(res.loc[i, 'qid'], res.loc[i, 'docno'], int(res.loc[i, 'rank']) + 1,
                                                res.loc[i, 'score']))
    with open(os.path.join(out_path,outname + '.res'), 'w') as f:
        f.writelines(out)

def main():
    out_path = 'sparseRetrieve/runs'
    outname = 'doc2query_bm25'
    qrels = '../../data/TRECCT2021/trec_2021_binarized_qrels.txt'

    qrels = rf.read_qrel(qrels)

    # cnt = 0
    # for qid in qrels:
    #     for doc in qrels[qid]:
    #         if int(qrels[qid][doc])>0:
    #             cnt += 1
    # print(cnt)

    pt.init(home_dir='/scratch/itee/s4575321/cache/')
    indexref = '../../data/TRECCT2021/pyterrier_json_cond'
    index = pt.IndexFactory.of(indexref)

    path_to_file = 'crossEncoder/data/doc2query_large.json'
    with open(path_to_file, 'r') as j:
        generated_queries = json.loads(j.read())

    path_to_file = '../../data/TRECCT2021/topics2021.xml'
    topics = rf.read_topics_ct21(path_to_file)

    queries = []
    for k, v in generated_queries.items():
        queries.append(' '.join(v))
    # for k, v in topics.items():
    #     queries.append(v)

    qids = [i for i in range(1, len(queries) + 1)]
    queries = [re.sub(r'[^A-Za-z0-9 ,.]', '', i) for i in queries]
    df_query = pd.DataFrame.from_dict({'qid': qids, 'query': queries})
    BM25_br = pt.BatchRetrieve(index, wmodel="BM25", metadata=['docno', 'text'], num_results=1000)
    res = BM25_br.transform(df_query)
    res.sort_values(by=['qid', 'score'], ascending=False)
    write_res(res, out_path, outname)
    eval(qrels, os.path.join(out_path, f'{outname}.res'), outname, out_path)
    eval(qrels, '../cttest/output_pyterrier_fusion_ct21/ielab-r2.res', 'ielabr2', out_path)


if __name__ == '__main__':
    main()
