#####################
# can be deleted when deploy
#####################
import argparse
import os
from crossEncoder.inference_e import eval

def combine_evals():
    root = './crossEncoder/eval'
    outpath = './crossEncoder/tmp_data/evals.txt'
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

if __name__ == '__main__':
    run_eval_for_all_res()
    # combine_evals()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test",default='True', choices=('True','False'))
    # args = parser.parse_args()
    # print(args.test, type(args.test))