import subprocess
import platform
import os

def read_resFile(path_to_result) -> list:
    assert path_to_result.strip().split(".")[-1] == 'res'

    res = {}
    with open(path_to_result, 'r') as f:
        contents = f.readlines()

    for line in contents:
        qid, _, docid, rank, score, name = line.strip().split("\t")
        if qid not in res:
            res[qid] = []
        res[qid].append(docid)
    return res

def eval_set_args(qrel, res):
    cmds = [['../trec_eval-9.0.7/trec_eval', '-m', 'map', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'Rprec', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'recip_rank', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'P.5', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'P.10', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'P.15', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'ndcg_cut.10', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'ndcg', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'recall.1000', qrel, res]]
    shell = platform.system() == "Windows"
    out = ''
    for cmd in cmds:
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=shell)
        stdout, stderr = process.communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        out += stdout.decode("utf-8")
    res = []
    for l in out.split('\n'):
        if len(l.split('\t')) == 3:
            name, _, value = l.split('\t')
            res.append((name.strip(), float(value.strip())))
    return res

def eval(qrelsFile, res_path, out_method_name, out_path):
    eval = eval_set_args(qrelsFile, res_path)

    cmd = '../trec_eval-9.0.7/trec_eval -q {} {} > {}'.format(qrelsFile, res_path,
                                                              os.path.join(out_path, out_method_name) + '.qeval')
    os.system(cmd)

    out_txt = 'name,' + ','.join([i[0] for i in eval]) + '\n'
    out_txt += out_method_name + ',' + ','.join([str(i[1]) for i in eval]) + '\n'
    with open(os.path.join(out_path, f'{out_method_name}.eval'), 'w') as f:
        f.writelines(out_txt)

def gen_res():
    modelname = 'biot5'
    path_to_qrels = '../../data/TRECCT2021/trec_2021_binarized_qrels.txt'
    path_to_file = f'denseRetrieve/data/ct21/ranking/biot5/{modelname}.txt'
    out_path = f'denseRetrieve/data/ct21/ranking/biot5/{modelname}.res'
    outdir_eval = 'denseRetrieve/data/ct21/evals'
    out = []
    cnt = 1
    cur_qid = None
    exist = set()
    for i in open(path_to_file, 'r'):
        qid, docid_dtype, score = i.split('\t')
        score = score.replace('\n','')
        if cur_qid != qid:
            exist = set()
            cur_qid = qid
            cnt = 1
        docid_dtype = 'NCT' + f'{int(docid_dtype):012}'
        docid, dtpye, pidx = docid_dtype[:11], docid_dtype[11:12], docid_dtype[12:]
        if docid not in exist:
            exist.add(docid)
            out.append(f"{qid}\tQ0\t{docid}\t{cnt}\t{score}\t{modelname}\n")
            cnt += 1
    with open(out_path, 'w') as f:
        f.writelines(out)
    eval(path_to_qrels, out_path, modelname, outdir_eval)

# if __name__ == '__main__':
#     gen_res()