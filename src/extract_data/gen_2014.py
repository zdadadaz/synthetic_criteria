import sys

sys.path.append('./../data_util.py')
from data_util import DataUtil
import csv
import os
from os import listdir
import xml.etree.ElementTree
from nltk.tokenize import sent_tokenize

def read_pkl():
    args = {'data_dir': 'i2b2/pkl_section', 'vocab_dir': 'i2b2/pkl_section', 'split_by_section': True}
    data = DataUtil(data_dir=args['data_dir'], vocab_dir=args['vocab_dir'],
                    split_by_sentence=not args['split_by_section'])
    data.load_split_data()
    print("i2b2 Training data: " + str(len(data.i2b2_train_data)))
    print("i2b2 Dev data: " + str(len(data.i2b2_dev_data)))
    print("i2b2 Test data: " + str(len(data.i2b2_test_data)))
    print(data.i2b2_dev_data)


class Load_i2b2:
    def __init__(self):
        self.run_all()
        self.all_class = ['NA', 'Chief complaint', 'Present illness', 'Medical history', 'Admission Medications',
                          'Allergies', 'Physical exam',
                          'Family history', 'Social history', 'Diagnosis', 'Findings', 'Treatment']
        self.required_class = ['Chief complaint', 'Present illness', 'Medical history', 'Admission Medications',
                               'Allergies', 'Physical exam',
                               'Family history', 'Social history', 'Diagnosis', 'Findings', 'Treatment']

    def get_offsets(self, file):
        # open annotations
        with open(file) as f:
            annotations = f.readlines()
            annotations.reverse()
            offsets = []
            for line in annotations:
                line.replace('\t\t', '\t')
                _, annotation, category = line.strip().split('\t')
                _, start, end = annotation.split(" ")

                offsets.append(start + "\t" + end + "\t" + category)
        return offsets

    def read_i2b2_single(self, target_ids, output_path, file_paths, annotation_path):

        labels_file = "./data/i2b2_labels_v1.4.csv"
        labels = {}

        with open(labels_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                labels[row[0].strip()] = row[1]

        with open(output_path, 'w') as outfile:
            outfile.write("id,header,text,labels\n")
            for patient_id in target_ids:
                for file_path in file_paths:
                    if os.path.isfile(file_path + '/' + patient_id):
                        file_name = file_path + "/" + patient_id
                e = xml.etree.ElementTree.parse(file_name).getroot()
                text = e.findall('TEXT')[0].text
                #  get all tag info
                pos2tags = {}
                for events in e.findall('TAGS')[0]:
                    if len(list(events)) == 0:
                        if events.tag == 'PHI':
                            continue
                        tmp_data = {'type': events.tag}
                        for key in events.attrib:
                            tmp_data[key] = events.get(key)
                        if 'start' in tmp_data:
                            pos2tags[tmp_data['start']] = tmp_data
                    else:
                        for anns in events:
                            tmp_data = {'type': anns.tag}
                            for key in anns.attrib:
                                tmp_data[key] = anns.get(key)
                            if 'start' in tmp_data:
                                pos2tags[tmp_data['start']] = tmp_data
                            break
                if os.path.isfile(annotation_path + '/Set1/' + patient_id[:-3] + "ann"):
                    annotation_path_dir = annotation_path + '/Set1'
                elif os.path.isfile(annotation_path + '/Set2/' + patient_id[:-3] + "ann"):
                    annotation_path_dir = annotation_path + '/Set2'
                else:
                    annotation_path_dir = annotation_path + '/Test'

                offsets = self.get_offsets(annotation_path_dir + "/" + patient_id[:-3] + "ann")
                offsets.append(str(len(text)) + "\tNA" + "\tNA")

                for i in range(len(offsets) - 1):
                    start, tmp_end, category = offsets[i].split("\t")
                    end, _, _ = offsets[i + 1].split("\t")
                    # dont save NA text
                    passage = text[int(start):int(end)]
                    # passage = labels[category.strip()] + ':'+passage[(int(tmp_end)-int(start)):]
                    sentences = sent_tokenize(passage)
                    passage = []
                    for s in sentences:
                        passage.append(s.replace('\n', ' ') + ' ')
                    passage = ''.join(passage)
                    mention_labels = []
                    for start_idx in pos2tags:
                        if int(start) <= int(start_idx) and int(start_idx) < int(end):
                            if pos2tags[start_idx]['type'] == 'SMOKER':
                                mention_labels.append(str((pos2tags[start_idx]['text'].replace(',',' '), pos2tags[start_idx]['type'].replace(',',' '),
                                                       pos2tags[start_idx]['status'].replace(',',' '))))
                            else:
                                type1 = pos2tags[start_idx]['type1'] if 'type1' in pos2tags[start_idx] else ''
                                mention_labels.append(str((pos2tags[start_idx]['text'], pos2tags[start_idx]['type'], type1)))
                    if category.strip() in labels and labels[category.strip()] == "NA" and not mention_labels:
                        continue
                    outfile.write('{},\"{}\",\"{}\",\"{}\"\n'.format(patient_id, labels[category.strip()], passage.replace('"',''),
                                                                     '|'.join(mention_labels).replace('"','').replace("'","")))
                    outfile.flush()

    def run_all(self):
        data_path = '../../data/n2c2/2014'
        train = data_path + "/training-RiskFactors-Complete-Set1"
        dev = data_path + "/training-RiskFactors-Complete-Set2"
        test = data_path + "/testing-RiskFactors-Complete"
        paths = [train, dev, test]
        output_path = 'i2b2/synth'

        target_ids = [f for f in listdir(train)]
        target_ids.extend([f for f in listdir(dev)])
        target_ids.extend([f for f in listdir(test)])

        print("target_ids", len(target_ids))
        self.read_i2b2_single(
            target_ids=target_ids,
            file_paths=paths,
            output_path=output_path + '/2014_parsed.csv',
            annotation_path='data/Dai_Section-Heading_Recognition_Corpus')


if __name__ == '__main__':
    Load_i2b2()
