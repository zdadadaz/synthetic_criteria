#####################
# can be deleted when deploy
#####################
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

if __name__ == '__main__':
    # run_eval_for_all_res()
    combine_evals()