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
    path_to_file = 'denseRetrieve/data/bioMSmarco/ranking/ance_rank.txt'
    out_path = 'denseRetrieve/data/bioMSmarco/ranking/ance_rank_trec.txt'
    out_dict = read_res(path_to_file)
    write_res(out_dict, out_path)

if __name__ == '__main__':
    tevatron_res_to_trec()
