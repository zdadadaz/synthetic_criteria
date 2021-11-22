import os
import subprocess
import platform


def eval_set_args(qrel, res):
    cmds = [['../trec_eval-9.0.7/trec_eval', '-m', 'map', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'Rprec', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'recip_rank', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'P.5', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'P.10', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'P.15', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'ndcg', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'recall', qrel, res]]
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
