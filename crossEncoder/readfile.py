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
