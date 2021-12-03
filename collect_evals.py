#####################
# can be deleted when deploy
#####################
import argparse
from src.utils.readfile import read_qrel
import os
# from src.utils.eval import eval_set_args
from sparseRetrieve.eval import eval_set_args
def eval(qrelsFile, res_path, out_method_name, out_path):
    eval = eval_set_args(qrelsFile, res_path)

    cmd = '../trec_eval-9.0.7/trec_eval -q {} {} > {}'.format(qrelsFile, res_path,
                                                              os.path.join(out_path, out_method_name) + '.qeval')
    os.system(cmd)

    out_txt = 'name,' + ','.join([i[0] for i in eval]) + '\n'
    out_txt += out_method_name + ',' + ','.join([str(i[1]) for i in eval]) + '\n'
    with open(os.path.join(out_path, f'{out_method_name}.eval'), 'w') as f:
        f.writelines(out_txt)

def combine_evals():
    # root = './crossEncoder/eval'
    root = 'sparseRetrieve/evals'
    outpath = './sparseRetrieve/tmp_data/evals.txt'
    out = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.split('.')[-1] == 'eval':
                for l in open(os.path.join(path, name)):
                    if not out:
                        out.append(l)
                    elif l.split(',')[0] != 'name':
                        out.append(l)
    with open(outpath, 'w') as f:
        f.writelines(out)

def run_eval_for_all_res():
    root = './crossEncoder/runs'
    path_to_qrel = '../../data/TRECCT2021/trec_2021_binarized_qrels.txt'
    outdir_eval = './crossEncoder/eval'
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.split('.')[-1] == 'res':
                outname = name.split('.')[0]
                res_path = os.path.join(path, name)
                eval(path_to_qrel, res_path, outname, outdir_eval)

def convert_res_to_trec_res():
    path = 'crossEncoder/runs/tc_medt5_3b_500_e.res'
    out_path = 'crossEncoder/runs/tc_medt5_3b_500_e_reform.res'
    out =[]
    for l in open(path):
        qid, _, docid, rank, score, method = l.strip().split('\t')
        out.append("{}\tQ0\t{}\t{}\t{}\t{}\n".format(qid, docid, rank, 1000-int(rank)+1, method))
    with open(out_path, 'w') as f:
        f.writelines(out)

def gen_judgement_doc():
    path_to_file = '../../data/test_collection/qrels-clinical_trials.tsv'
    out_path = 'data/judgment/ct2016_judgement.res'
    # path_to_file = '../../data/TRECCT2021/trec-ct2021-qrels.txt'
    # out_path = 'data/judgment/ct2021_judgement.res'
    out = []
    qrels = read_qrel(path_to_file)
    for qid in qrels:
        cnt = 1
        for doc in qrels[qid]:
            out.append(f"{qid}\tQ0\t{doc}\t{cnt}\t{qrels[qid][doc]}\tmethod\n")
            cnt += 1
    with open(out_path, 'w') as f:
        f.writelines(out)

if __name__ == '__main__':
    # run_eval_for_all_res()
    # combine_evals()
    gen_judgement_doc()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test",default='True', choices=('True','False'))
    # args = parser.parse_args()
    # print(args.test, type(args.test))