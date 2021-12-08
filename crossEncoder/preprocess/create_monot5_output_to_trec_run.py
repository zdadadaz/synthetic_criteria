"""
This script convert monoT5 output file to msmarco run file
"""
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--t5_output", type=str, required=True,
                    help="tsv file with two columns, <label> and <score>")
parser.add_argument("--t5_output_ids", type=str, required=True,
                    help="tsv file with two columns, <query_id> and <doc_id>")
parser.add_argument("--mono_run", type=str, required=True,
                    help="path to output mono run, tsv file, with <query_id>, <doc_id> and <rank>")
parser.add_argument("--choose_type", type=str, default='e', help="choose e, d type for evaluation")
args = parser.parse_args()

def main():
    examples = collections.defaultdict(list)
    with open(args.t5_output_ids) as f_gt, open(args.t5_output) as f_pred:
        for line_gt, line_pred in zip(f_gt, f_pred):
            query_id, doc_id_type = line_gt.strip().split('\t')
            doc_id, dtype, pidx= doc_id_type.split('_')
            if dtype == args.choose_type:
                _, score = line_pred.strip().split('\t')
                score = float(score)
                examples[query_id].append((score, doc_id, dtype, pidx))

    with open(args.mono_run, 'w') as fout:
        for query_id, doc_ids_scores in examples.items():
            doc_ids_scores.sort(key=lambda x: x[0], reverse=True)
            exist = set()
            rank = 1
            for (score, doc_id, _, _) in doc_ids_scores:
                if doc_id not in exist:
                    fout.write(f'{query_id}\tQ0\t{doc_id}\t{rank}\t{score}\tmethod\n')
                    exist.add(doc_id)
                    rank += 1

if __name__ == '__main__':
    main()

