import pandas as pd
import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
import numpy as np
from os import listdir
import os
import xml.etree.ElementTree
import json
import re


class Load_i2b2:
    def __init__(self, offNA, output_dir, outputFileName, path_to_dir):
        self.regex_script = r'^[0-9]+ *[\.)]|^[a-z]? *[\.)]|[0-9]+[:|.][0-9]+|[0-9]+\+'
        self.offNA = offNA
        self.output_dir = output_dir
        self.outputFileName = outputFileName
        self.run_all(path_to_dir)
        self.write_diagnosis(output_dir, '2012')

    def run_all(self, path_to_dir):
        bert_model = 'bert-base-uncased'
        num_labels = 12
        self.init_model(bert_model, num_labels)
        self.init_tokenizer(bert_model)

        data = self.read_files(path_to_dir)
        eval_examples = self.gen_example(data)
        tokenized = self.tokenize(eval_examples)
        logits = self.run_section_classification(tokenized)
        self.combine_result(data, eval_examples, logits)

    def init_tokenizer(self, bert_model):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

    def init_model(self, bert_model, num_labels):
        self.max_seq_length = 128
        self.eval_batch_size = 8
        self.label_list = ['NA', 'Chief complaint', 'Present illness', 'Medical history', 'Admission Medications',
                           'Allergies', 'Physical exam', 'Family history', 'Social history', 'Diagnosis', 'Findings',
                           'Treatment']
        pytorch_model = 'output_bert_section/pytorch_model.bin'
        # load model
        checkpoint = torch.load(pytorch_model)
        self.model = BertForSequenceClassification.from_pretrained(bert_model,
                                                                   num_labels=num_labels)
        self.model.load_state_dict(checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def read_files(self, path_to_dir):
        extract_events = ['PROBLEM', 'TREATMENT']
        out = {}
        for dirPath in ['ground_truth/merged_i2b2/', '2012-07-15.original-annotation.release/']:
            patient_ids = [i[:-4] for i in listdir(os.path.join(path_to_dir, dirPath)) if i.split('.')[-1] == 'txt']
            for patient_id in patient_ids:
                out[patient_id] = {'text': None, 'event': None, 'group': []}
                filePath = os.path.join(path_to_dir, dirPath + patient_id + '.txt')
                if os.path.exists(filePath):
                    with open(filePath, 'r') as f:
                        text = f.read()
                        out[patient_id]['text'] = text.split('\n')
                        cur = -1
                        for idx, l in enumerate(out[patient_id]['text']):
                            if ':' in l and '#' not in l and '::' not in l and l.strip()[
                                -1] != ',' and ':*' not in l and not re.search(self.regex_script, l.strip()):
                                if cur > -1:
                                    out[patient_id]['group'].append((cur, idx))
                                cur = idx
                        if out[patient_id]['group'] and out[patient_id]['group'][-1][0] != cur:
                            out[patient_id]['group'].append((cur, idx))
                else:
                    raise ValueError("No exist file " + filePath)
                eventPath = os.path.join(path_to_dir, dirPath + patient_id + '.extent')
                if os.path.exists(eventPath):
                    out[patient_id]['event'] = [[] for i in range(len(out[patient_id]['text']))]
                    with open(eventPath, 'r') as f:
                        events = f.readlines()
                    for li in events:
                        if 'EVENT' != li[:5]:
                            continue
                        tags = li.split('||')
                        if tags[1][6:-1] not in extract_events:
                            continue
                        line_idx = int(tags[0].split(' ')[-1].split(':')[0]) - 1
                        outdata = (
                        ' '.join(tags[0].replace('"', '')[6:].split(' ')[:-2]), tags[1][6:-1], tags[3][10:-2])
                        out[patient_id]['event'][line_idx].append(outdata)
                else:
                    del out[patient_id]
                    print("No exist file " + eventPath)
                    # raise ValueError("No exist file " + eventPath)
        return out

    def tokenize(self, examples):
        out = []
        for id, text in examples:
            tokens_a = self.tokenizer.tokenize(text)
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            out.append((input_ids, input_mask, segment_ids))
        return out

    def gen_example(self, data):
        out = []
        for id in data.keys():
            for idx, j in enumerate(data[id]['group']):
                out.append((id + '--' + str(idx), ' '.join(data[id]['text'][j[0]:j[1]])))
        return out

    def run_section_classification(self, eval_examples):
        all_input_ids = torch.tensor([e[0] for e in eval_examples], dtype=torch.long)
        all_input_mask = torch.tensor([e[1] for e in eval_examples], dtype=torch.long)
        all_segment_ids = torch.tensor([e[2] for e in eval_examples], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        self.model.eval()
        all_logits = []

        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            sorted_logits = np.argsort(logits)
            all_logits.extend(sorted_logits)
        return all_logits

    def combine_result(self, data, eval_examples, logits):
        with open(os.path.join(self.output_dir, self.outputFileName), 'w') as fout:
            fout.write("id,header,text,labels\n")
            for idx, (p_id, e) in enumerate(eval_examples):
                patient_id, pss_id = p_id.split('--')
                g = data[patient_id]['group'][int(pss_id)]
                out_events = [re.sub(r'[\"\']','',str(j)) for i in
                              data[patient_id]['event'][int(g[0]):int(g[1])] if i for j in i]
                if self.offNA and self.label_list[logits[idx][-1]] == 'NA' and not out_events:
                    continue
                # if ' date ' in e.lower() and not out_events:
                #     continue
                fout.write(
                    '{},\"{}\",\"{}\",\"{}\"\n'.format(patient_id, self.label_list[logits[idx][-1]], e.replace('"', ''),
                                                       '|'.join(out_events)))
                fout.flush()

    def write_diagnosis(self, output_dir, year):
        df = pd.read_csv(os.path.join(output_dir, f'{year}_parsed.csv'))
        outdict = {}
        for i in range(len(df)):
            id = str(df.iloc[i, 0])
            if id not in outdict:
                outdict[id] = []
            if df.iloc[i, 1] == 'Diagnosis' or df.iloc[i, 1] == 'Treatment':
                outdict[id].append(df.iloc[i, 2])

        json_object = json.dumps(outdict, indent=2)
        with open(os.path.join(output_dir, f'{year}_docDiagnosis.json'), "w") as outfile:
            outfile.write(json_object)


class Load_i2b2_2011(Load_i2b2):
    def __init__(self, offNA, output_dir, outputFileName, path_to_dir):
        super().__init__(offNA, output_dir, outputFileName, path_to_dir)
        self.write_diagnosis(output_dir, '2011')

    def read_files(self, path_to_dir):
        extract_events = ['problem', 'treatment']
        out = {}
        for dirPath in ['Beth_Train/', 'Partners_Train/', 'i2b2_Test/i2b2_Beth_Test/', 'i2b2_Test/i2b2_Partners_Test/']:
            patient_ids = [i[:-4] for i in listdir(os.path.join(path_to_dir, dirPath + 'docs')) if
                           i.split('.')[-1] == 'txt']
            for patient_id in patient_ids:
                out[patient_id] = {'text': None, 'event': None, 'group': []}
                filePath = os.path.join(path_to_dir, dirPath + 'docs/' + patient_id + '.txt')
                if os.path.exists(filePath):
                    with open(filePath, 'r') as f:
                        text = f.read()
                        out[patient_id]['text'] = text.split('\n')
                        cur = -1
                        for idx, l in enumerate(out[patient_id]['text']):
                            if ':' in l and '#' not in l and '::' not in l and l.strip()[
                                -1] != ',' and ':*' not in l and not re.search(self.regex_script, l.strip()):
                                if cur > -1:
                                    out[patient_id]['group'].append((cur, idx))
                                cur = idx
                        if out[patient_id]['group'] and out[patient_id]['group'][-1][0] != cur:
                            out[patient_id]['group'].append((cur, idx))
                else:
                    raise ValueError("No exist file " + filePath)
                eventPath = os.path.join(path_to_dir, dirPath + 'concepts/' + patient_id + '.txt.con')
                if os.path.exists(eventPath):
                    out[patient_id]['event'] = [[] for i in range(len(out[patient_id]['text']))]
                    with open(eventPath, 'r') as f:
                        events = f.readlines()
                    for li in events:
                        tags = li.split('||')
                        if tags[-1][3:-2].lower() not in extract_events:
                            continue
                        line_idx = int(tags[0].split(' ')[-1].split(':')[0]) - 1
                        outdata = (' '.join(tags[0].replace('"', '').split(' ')[0:-2])[2:], tags[1][3:-2])
                        out[patient_id]['event'][line_idx].append(outdata)
                else:
                    del out[patient_id]
                    print("No exist file " + eventPath)
        return out


class Load_i2b2_2010(Load_i2b2):
    def __init__(self, offNA, output_dir, outputFileName, path_to_dir):
        super().__init__(offNA, output_dir, outputFileName, path_to_dir)
        self.write_diagnosis(output_dir, '2010')

    def read_files(self, path_to_dir):
        extract_events = ['problem', 'treatment']
        out = {}
        for dirPath in ['concept_assertion_relation_training_data/beth/',
                        'concept_assertion_relation_training_data/partners/',
                        'test_data/']:
            if 'test_data' in dirPath:
                patient_ids = [i[:-4] for i in listdir(os.path.join(path_to_dir, dirPath)) if
                               i.split('.')[-1] == 'txt']
            else:
                patient_ids = [i[:-4] for i in listdir(os.path.join(path_to_dir, dirPath + 'txt')) if
                               i.split('.')[-1] == 'txt']
            for patient_id in patient_ids:
                out[patient_id] = {'text': None, 'event': None, 'group': []}
                prefix = '' if 'test_data' in dirPath else 'txt/'
                filePath = os.path.join(path_to_dir, dirPath + prefix + patient_id + '.txt')
                if os.path.exists(filePath):
                    with open(filePath, 'r') as f:
                        text = f.read()
                        out[patient_id]['text'] = text.split('\n')
                        cur = -1
                        for idx, l in enumerate(out[patient_id]['text']):
                            if ':' in l and '#' not in l and '::' not in l and l.strip()[
                                -1] != ',' and ':*' not in l and not re.search(self.regex_script, l.strip()):
                                if cur > -1:
                                    out[patient_id]['group'].append((cur, idx))
                                cur = idx
                        if out[patient_id]['group'] and out[patient_id]['group'][-1][0] != cur:
                            out[patient_id]['group'].append((cur, idx))
                else:
                    raise ValueError("No exist file " + filePath)
                if 'test_data' in dirPath:
                    eventPath = os.path.join(path_to_dir,
                                             'reference_standard_for_test_data/' + 'concepts/' + patient_id + '.con')
                    astPath = os.path.join(path_to_dir,
                                           'reference_standard_for_test_data/' + 'ast/' + patient_id + '.ast')
                else:
                    eventPath = os.path.join(path_to_dir, dirPath + 'concept/' + patient_id + '.con')
                    astPath = os.path.join(path_to_dir, dirPath + 'ast/' + patient_id + '.ast')
                if os.path.exists(eventPath):
                    out[patient_id]['event'] = [[] for i in range(len(out[patient_id]['text']))]
                    with open(eventPath, 'r') as f:
                        events = f.readlines()
                    concept2ast = {}
                    for l in open(astPath, 'r'):
                        concept = l.split('"')[1]
                        status = l.split('"')[-1]
                        concept2ast[concept] = 'POS' if status != 'absent' else 'NEG'

                    for li in events:
                        tags = li.split('||')
                        if tags[-1][3:-2].lower() not in extract_events:
                            continue
                        line_idx = int(tags[0].split(' ')[-1].split(':')[0]) - 1
                        m = re.search(r'(\".*\")', tags[0])
                        concept = m.group(1).strip('"')
                        ispos = 'POS' if concept not in concept2ast else concept2ast[concept]
                        outdata = (
                        ' '.join(tags[0].replace('"', '').split(' ')[0:-2])[2:], tags[1][3:-2], ispos)
                        out[patient_id]['event'][line_idx].append(outdata)
                else:
                    del out[patient_id]
                    print("No exist file " + eventPath)
        return out


class Load_i2b2_2009(Load_i2b2):
    def __init__(self, offNA, output_dir, outputFileName, path_to_dir):
        super().__init__(offNA, output_dir, outputFileName, path_to_dir)
        self.write_diagnosis(output_dir, '2009')

    def read_files(self, path_to_dir):
        out = {}
        for dirPath in ['training.sets.released/' + str(i) + '/' for i in range(1, 11)] + [
            'train.test.released.8.17.09/']:
            patient_ids = [i for i in listdir(os.path.join(path_to_dir, dirPath))]
            for patient_id in patient_ids:
                out[patient_id] = {'text': None, 'event': None, 'group': []}
                filePath = os.path.join(path_to_dir, dirPath + patient_id)
                if os.path.exists(filePath):
                    with open(filePath, 'r') as f:
                        text = f.read()
                        out[patient_id]['text'] = text.split('\n')
                        cur = -1
                        for idx, l in enumerate(out[patient_id]['text']):
                            if ':' in l and '#' not in l and '::' not in l and l.strip()[
                                -1] != ',' and ':*' not in l and not re.search(self.regex_script, l.strip()):
                                if cur > -1:
                                    out[patient_id]['group'].append((cur, idx))
                                cur = idx
                        if out[patient_id]['group'] and out[patient_id]['group'][-1][0] != cur:
                            out[patient_id]['group'].append((cur, idx))
                else:
                    raise ValueError("No exist file " + filePath)
                out[patient_id]['event'] = [[] for i in range(len(out[patient_id]['text']))]
        return out


class Load_i2b2_2008(Load_i2b2):
    def __init__(self, offNA, output_dir, outputFileName, path_to_dir):
        super().__init__(offNA, output_dir, outputFileName, path_to_dir)
        self.write_diagnosis(output_dir, '2008')

    def read_files(self, path_to_dir):
        out = {}
        outJudge = {}
        for dirPath in ['obesity_patient_records_training.xml', 'obesity_patient_records_training2.xml',
                        'obesity_patient_records_test.xml']:
            e = xml.etree.ElementTree.parse(os.path.join(path_to_dir, dirPath)).getroot()
            for doc in e.findall('docs')[0]:
                patient_id = doc.get('id')
                out[patient_id] = {'text': None, 'event': None, 'group': []}
                for content in doc:
                    text = content.text
                    out[patient_id]['text'] = text.split('\n')
                    cur = -1
                    if patient_id == '2':
                        a = 1
                    for idx, l in enumerate(out[patient_id]['text']):
                        if ':' in l and '#' not in l and '::' not in l and l.strip()[
                            -1] != ',' and ':*' not in l and not re.search(self.regex_script, l.strip()):
                            if cur > -1:
                                out[patient_id]['group'].append((cur, idx))
                            cur = idx
                    if out[patient_id]['group'] and out[patient_id]['group'][-1][0] != cur:
                        out[patient_id]['group'].append((cur, idx))
                out[patient_id]['event'] = [[] for i in range(len(out[patient_id]['text']))]
        for prefix in ['training', 'test']:
            annoPath = 'obesity_standoff_annotations_{}.xml'.format(prefix)
            e = xml.etree.ElementTree.parse(os.path.join(path_to_dir, annoPath)).getroot()
            for diseases in e:
                anno_type = diseases.get('source')
                for disease in diseases:
                    for doc in disease:
                        if doc.get('judgment') == 'Y':
                            patient_id = doc.get('id')
                            if patient_id not in outJudge:
                                outJudge[patient_id] = {'intuitive': [], 'textual': []}
                            outJudge[patient_id][anno_type].append(disease.get('name'))
        json_object = json.dumps(outJudge, indent=2)
        with open(os.path.join(self.output_dir, '2008_docJudgement.json'), "w") as outfile:
            outfile.write(json_object)
        return out


class Load_i2b2_2006(Load_i2b2):
    def __init__(self, offNA, output_dir, outputFileName, path_to_dir):
        super().__init__(offNA, output_dir, outputFileName, path_to_dir)
        self.write_diagnosis(output_dir, '2006')

    def read_files(self, path_to_dir):
        # < [ ^ <]+? > * < [ ^ <]+? >
        # re.sub('<[^<]+?>', '', text)

        out = {}
        outJudge = {}
        for dirPath in ['smokers_surrogate_train_all_version2.xml',
                        'smokers_surrogate_test_all_groundtruth_version2.xml']:
            e = xml.etree.ElementTree.parse(os.path.join(path_to_dir, dirPath)).getroot()
            for doc in e.findall('RECORD'):
                patient_id = doc.get('ID')
                Smoking, content = doc
                out[patient_id] = {'text': None, 'event': None, 'group': []}
                text = content.text
                out[patient_id]['text'] = text.split('\n')
                cur = -1
                if patient_id == '681':
                    a = 1
                for idx, l in enumerate(out[patient_id]['text']):
                    if ':' in l and '#' not in l and '::' not in l and l.strip()[
                        -1] != ',' and ':*' not in l and not re.search(self.regex_script, l.strip()):
                        if cur > -1:
                            out[patient_id]['group'].append((cur, idx))
                        cur = idx
                if out[patient_id]['group'] and out[patient_id]['group'][-1][0] != cur:
                    out[patient_id]['group'].append((cur, idx))
                out[patient_id]['event'] = [[] for i in range(len(out[patient_id]['text']))]
                outJudge[patient_id] = Smoking.get('STATUS')
        json_object = json.dumps(outJudge, indent=2)
        with open(os.path.join(self.output_dir, '2006_docJudgement.json'), "w") as outfile:
            outfile.write(json_object)
        return out


if __name__ == '__main__':
    outputDir = 'i2b2/synth'
    # passage-based label
    print('Generate 2012')
    Load_i2b2(True, outputDir, '2012_parsed.csv', '../../data/n2c2/2012')
    print('Generate 2011')
    Load_i2b2_2011(True, outputDir, '2011_parsed.csv', '../../data/n2c2/2011')
    print('Generate 2010')
    Load_i2b2_2010(True, outputDir, '2010_parsed.csv', '../../data/n2c2/2010')

    # no label(only have medications)
    print('Generate 2009')
    Load_i2b2_2009(True, outputDir, '2009_parsed.csv', '../../data/n2c2/2009')

    # document-based label
    print('Generate 2008')
    Load_i2b2_2008(True, outputDir, '2008_parsed.csv', '../../data/n2c2/2008')
    print('Generate 2006')
    Load_i2b2_2006(True, outputDir, '2006_parsed.csv', '../../data/n2c2/2006')
