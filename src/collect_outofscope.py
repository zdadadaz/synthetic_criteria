import json

def collect_outofscope():
    path_to_file = '../ctCriteria/utils_parse_cfg/anno/query_assignment.json.json'
    path_out = 'data/template/outofscope.txt'

    with open(path_to_file, 'r') as j:
        alldoc = json.loads(j.read())

    out = set()
    for doc in alldoc:
        for c in doc['criteria']:
            if c['type_id'] == 0:
                out.add(c['text'] + '\n')
    out = [str(idx+1) + '\t' + i for idx, i in enumerate(out)]
    with open(path_out, 'w') as f:
        f.writelines(out)

if __name__ == '__main__':
    collect_outofscope()