import re
import json
import os
from pyserini.search import SimpleSearcher
from eval import eval_set_args
import readfile as rf
from collections import defaultdict

def RRF_fusion(res_dir, qids, path_out, method_name):
    filelist = []
    for path, subdirs, files in os.walk(res_dir):
        for name in files:
            if name.split('.')[-1] == 'res':
                filelist.append(rf.read_resFile(os.path.join(path, name)))
    out = []
    for qid in qids:
        doc2score = defaultdict(int)
        for file in filelist:
            for idx, docid in enumerate(file[qid]):
                doc2score[docid] += 1/(60 + idx + 1)
        ranklist = [(k,v) for k, v in doc2score.items()]
        ranklist.sort(key=lambda k:k[1], reverse=True)
        for idx, r in enumerate(ranklist[:1000]):
            out.append("{}\tQ0\t{}\t{}\t{}\t{}\n".format(qid, r[0], idx+1, r[1], method_name))
    with open(path_out, 'w') as f:
        f.writelines(out)

def eval(qrelsFile, res_path, out_method_name, out_path):
    eval = eval_set_args(qrelsFile, res_path)

    cmd = '../trec_eval-9.0.7/trec_eval -c -q -l 2 {} {} > {}'.format(qrelsFile, res_path,
                                                              os.path.join(out_path, out_method_name) + '.qeval')
    os.system(cmd)

    out_txt = 'name,' + ','.join([i[0] for i in eval]) + '\n'
    out_txt += out_method_name + ',' + ','.join([str(i[1]) for i in eval]) + '\n'
    with open(os.path.join(out_path, f'{out_method_name}.eval'), 'w') as f:
        f.writelines(out_txt)


def write_hits(hits, output_path, bm25_k=1000, excludeZero=False, run_name='noName'):
    trec = []
    for qid in hits.keys():
        cnt = 1
        for hit in hits[qid]:
            if cnt > bm25_k:
                break
            if (not excludeZero) or (hit.score > 0.0001):
                trec.append(str(qid) + "\tQ0\t" + str(hit.docid) + "\t" + str(cnt) + "\t" + str(
                    hit.score) + "\t" + run_name + "\n")
                cnt += 1
    trec.sort(key=lambda k: int(k.split("\t")[0]))
    with open(output_path + '.res', "w") as f:
        f.writelines(trec)


def write_res(res, out_path, outname):
    out = []
    for i in range(len(res)):
        out.append(
            "{}\tQ0\t{}\t{}\t{}\truns\n".format(res.loc[i, 'qid'], res.loc[i, 'docno'], int(res.loc[i, 'rank']) + 1,
                                                res.loc[i, 'score']))
    with open(os.path.join(out_path, outname + '.res'), 'w') as f:
        f.writelines(out)


def main():
    output_path = 'sparseRetrieve/runs'
    intermittent_path = 'sparseRetrieve/runs/intermittent'
    out_eval_path = 'sparseRetrieve/evals'
    qrelsPath = '../../data/TRECCT2021/trec-ct2021-qrels.txt'
    method_name = 'doc2query_bm25rm_fusion'
    path_out = os.path.join(output_path, method_name + '.res')

    qrels = rf.read_qrel(qrelsPath)
    searcher = SimpleSearcher('../pyserini/indexes/TRECCT2021')
    searcher.set_rm3()

    path_to_file = 'crossEncoder/data/doc2query_large.json'
    with open(path_to_file, 'r') as j:
        generated_queries = json.loads(j.read())

    path_to_file = '../../data/TRECCT2021/topics2021.xml'
    topics = rf.read_topics_ct21(path_to_file)

    qids = [str(i+1) for i in range(len(topics))]
    for idx in range(0, 41):
        outname = 'bm25rm' if idx == 0 else 'doc2query_bm25rm'
        queries = []
        if idx == 0:
            for k, v in topics.items():
                queries.append(v)
        else:
            for k, v in generated_queries.items():
                queries.append(v[idx-1])
        queries = [re.sub(r'[^A-Za-z0-9 ,.]', '', q) for q in queries]
        hits = {}
        for pidx, q in enumerate(queries):
            q = re.sub(r'[^A-Za-z0-9 ,.]', '', q)
            hit = searcher.search(q, k=1000)
            hits[pidx + 1] = hit
        out_path = os.path.join(intermittent_path, 'pyserini_{}_{}'.format(outname, idx))
        write_hits(hits, out_path, 1000, run_name=str(idx))

    RRF_fusion(intermittent_path, qids, path_out, method_name)
    eval(qrelsPath, path_out, method_name, out_eval_path)
    eval(qrelsPath, intermittent_path + '/pyserini_bm25rm_0.res', 'bm25rm', out_eval_path)
    # eval('../../data/TRECCT2021/trec-ct2021-qrels.txt', 'crossEncoder/data/ielab-r2.res', 'ielab-r2', out_eval_path)


if __name__ == '__main__':
    main()
