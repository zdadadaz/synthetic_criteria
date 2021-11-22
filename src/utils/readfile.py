from collections import defaultdict
import xml.etree.ElementTree as ET

def read_topics_ct21(path_to_topics) -> dict:
    '''
    return a dict that maps qid, content pair
    '''
    topics = defaultdict(dict)
    tree = ET.parse(path_to_topics)
    root = tree.getroot()
    for topic in root:
        idx = topic.attrib['number']
        topics[idx] = topic.text
    return topics

def read_qrel(path_to_qrel) -> dict:
    """
    return a dictionary that maps qid, docid pair to its relevance label.
    """
    qrel = {}
    with open(path_to_qrel, 'r') as f:
        contents = f.readlines()

    for line in contents:
        if path_to_qrel.strip().split(".")[-1] == 'txt':
            qid, _, docid, rel = line.strip().split(" ")
        elif path_to_qrel.strip().split(".")[-1] == 'tsv':
            qid, _, docid, rel = line.strip().split("\t")
        if qid in qrel.keys():
            pass
        else:
            qrel[qid] = {}
        qrel[qid][docid] = int(rel)

    return qrel

def read_ts_topic(path_to_topics):
    with open(path_to_topics, 'r') as f:
        contents = f.readlines()

    topic_dict = {}
    cur_topic_num = None
    for line in contents:
        if "<NUM>" in line:
            cur_topic_num = line.strip().split('NUM>')[1][:-2]
            if cur_topic_num not in topic_dict:
                topic_dict[cur_topic_num] = ''

        if "<TITLE>" in line:
            topic_dict[cur_topic_num] += ' ' + line.strip().split('<TITLE>')[1]
    return topic_dict

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