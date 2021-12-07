import pandas as pd
import os
import random
import string
import json
import pyterrier as pt
import re
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

class Mimiciii:
    def __init__(self):
        indexref = '../../data/TRECCT2021/pyterrier_criteria'
        self.index = pt.IndexFactory.of(indexref)
        self.generate_samples('DIA', 'disease')
        # self.generate_samples('PRO', 'treatment')

    def generate_samples(self, typeDB, template_type):
        if template_type == 'disease':
            out_path_dia = 'output/mimic/mimic_dia_8neg.json'
        else:
            out_path_dia = 'output/mimic/mimic_pro_8neg.json'
        df1, icd9 = self.read_file(typeDB)
        template = self.read_template()
        synth_data = self.gen_synthesis(df1, icd9, template, template_type)
        self.write_out(synth_data, out_path_dia)

    def read_file(self, isDia='DIA'):
        sufix = 'DIAGNOSES' if isDia == 'DIA' else 'PROCEDURES'
        icd9 = pd.read_csv(f'../../data/mimicIII/full/D_ICD_{sufix}.csv', dtype={"ICD9_CODE": str})
        icd9["NAMES"] = icd9.LONG_TITLE.str.replace('[{}]'.format(string.punctuation),
                                                    '').str.lower().str.replace('unspecified', '').str.split()
        icd9["NAMES"] = icd9.NAMES.apply(lambda x: " ".join([word for word in x]))
        icd9 = icd9.set_index('ICD9_CODE').T.to_dict('list')
        for type in ['train', 'test', 'val']:
            path_to_file = f'../ehr_section_prediction/mimicIII/{isDia}_PLUS_{type}.csv'
            if type == 'train':
                df = pd.read_csv(path_to_file)
            else:
                df.append(pd.read_csv(path_to_file))
        return df, icd9

    def read_template(self):
        path_to_dir = 'data/template'
        out = {}
        for i in ['admindrug', 'allergy', 'disease', 'outofscope', 'patienthistory', 'treatment']:
            out[i] = []
            for l in open(os.path.join(path_to_dir, i + '.txt')):
                _, p = l.strip().split('\t')
                out[i].append(p)
        return out

    def gen_synthesis(self, df, icd9, template, template_type):
        # can use icd9 three digit code for general disease name (follow up)
        out = []
        allcode = list(icd9.keys())
        tot = 0
        for i in tqdm(range(len(df))):
            dias = [None] * 3
            exists = set()
            pid, text, dias[0], code_short, dias[1], _, dias[2] = df.iloc[i]
            out_tmp = {'id': str(pid), 'pos': [], 'neg': [], 'ret_pos': [], 'ret_neg': []}

            code_short = set(code_short.split('|'))
            # positive
            for dia in dias:
                if str(dia) == 'nan':
                    continue
                tmp = dia.split('|')
                for idx, d in enumerate(tmp):
                    if d.strip() in exists or len(d.strip()) == 0:
                        continue
                    exists.add(d.strip())
                    rand = random.randint(1, len(template[template_type]) - 1)
                    out_tmp['pos'].append(template[template_type][rand].replace('[mask]', d))

            # negative
            tried = set()
            while len(out_tmp['neg']) < min(len(out_tmp['pos']) * 8, 100) and len(tried) != len(allcode):
                code_tmp = random.choice(allcode)
                tried.add(code_tmp)
                if code_tmp.startswith('V') or code_tmp.startswith('E'):
                    code_tmp_short = code_tmp[:4]
                else:
                    code_tmp_short = code_tmp[:3]
                if code_tmp_short not in code_short and code_tmp not in exists:
                    rand = random.randint(1, len(template[template_type]) - 1)
                    out_tmp['neg'].append(template[template_type][rand].replace('[mask]', icd9[code_tmp][-1]))
                    exists.add(code_tmp)
            out_tmp['ret_pos'] = self.simple_search(out_tmp['pos'])
            out_tmp['ret_neg'] = self.simple_search(out_tmp['neg'])

            tot += len(out_tmp['pos']) * 2
            out.append(out_tmp)

        print('total samples', tot)
        return out

    def simple_search(self, queries):
        qids = [i for i in range(len(queries))]
        queries = [re.sub(r'[^A-Za-z0-9 ,.]', '', i) for i in queries]
        df_query = pd.DataFrame.from_dict({'qid': qids, 'query': queries})
        BM25_br = pt.BatchRetrieve(self.index, wmodel="BM25", metadata=['docno', 'text'], num_results=1)
        res = BM25_br.transform(df_query)
        res.sort_values(by=['qid', 'score'], ascending=False)
        return res['text'].to_list()

    def write_out(self, out, out_path):
        if not os.path.exists('/'.join(out_path.split('/')[:-1])):
            os.makedirs('/'.join(out_path.split('/')[:-1]), exist_ok=True)
        json_object = json.dumps(out, indent=2)
        with open(out_path, "w") as outfile:
            outfile.write(json_object)


class i2b2_doc_2009(Mimiciii):
    def __init__(self, year):
        indexref = '../../data/TRECCT2021/pyterrier_criteria'
        self.index = pt.IndexFactory.of(indexref)
        self.generate_samples(year)

    def generate_samples(self, year='2009'):
        out_path_dia = f'output/i2b2/{year}_dia_8neg.json'
        df1, icd9 = self.read_file(year)
        template = self.read_template()
        synth_data = self.gen_synthesis(df1, icd9, template, {'d': 'disease'}, ['d'])
        self.write_out(synth_data, out_path_dia)

    def read_file(self, year):
        path_to_file = f'../ehr_section_prediction/i2b2/synth/{year}_parsed.csv'
        path_to_diagnose = f'../ehr_section_prediction/i2b2/synth/{year}_docDiagnosis.json'
        df = pd.read_csv(path_to_file)
        df = df.dropna(how='any', subset=['text'], axis=0)
        df = df.drop(df[(df['header'] == 'Diagnosis') | (df['header'] == 'Treatment')].index)
        df = df.groupby(['id'])['text'].apply(" ".join).reset_index()

        exclusion_disease = set(['infections', 'complications', 'conditions'])
        replace_diag_list = ['PRINCIPAL DISCHARGE DIAGNOSIS ;Responsible After Study for Causing Admission )',
                             'ADMIT DIAGNOSIS:', 'ADMIT DIAGNOSIS', 'PRINCIPAL DISCHARGE DIAGNOSIS:',
                             'PRINCIPAL DISCHARGE DIAGNOSIS', 'secondary diagnoses',
                             'OTHER DIAGNOSIS:', 'OTHER DIAGNOSIS', 'ADMISSION DIAGNOSES:', 'ADMISSION DIAGNOSES',
                             'PREOPERATIVE DIAGNOSES:', 'PREOPERATIVE DIAGNOSES', 'principal diagnosis:',
                             'principal diagnosis', 'discharge diagnosis:', 'affecting treatment/stay',
                             'discharge diagnosis', 'affecting treatment / stay']
        replace_diag_list = [i.lower() for i in replace_diag_list]
        replace_proc_list = ['PROCEDURE:', 'PROCEDURES:', 'PROCEDURE PERFORMED:', "principal procedure or operation",
                             "associated procedures or operations", "postpartum therapeutic procedures", ]
        exclusion_procedure = set(['procedures', 'complications', 'conditions', 'none'])

        label = {}
        if os.path.exists(path_to_diagnose):
            with open(path_to_diagnose, 'r') as j:
                alldoc = json.loads(j.read())

            for k, v in alldoc.items():
                label[k] = {'d': [], 'p': []}
                exist = set()
                for i in v:
                    i = i.lower()
                    if 'diagnos' in i:
                        for j in replace_diag_list:
                            i = i.replace(j, ',')
                        i = re.sub(r'[0-9]+ *[\.)]', ',', i)
                        for d in i.split(','):
                            if not d:
                                continue
                            if ')' in d:
                                for dd in d.split(')'):
                                    if '(' in dd:
                                        abbr, org_name = dd.split('(')
                                        if abbr.strip() != org_name.strip():
                                            label[k]['d'].append(abbr.strip() + ' (' + org_name.strip() + ')')
                                        else:
                                            abbr = re.sub('[\';:,.)(\]]', '', abbr)
                                            if abbr and abbr not in exist:
                                                if abbr.strip() not in exclusion_disease:
                                                    label[k]['d'].append(abbr.strip())
                                                    exist.add(abbr.strip())
                            else:
                                d = re.sub('[\';:,.)(\]]', '', d)
                                d = d.strip()
                                if d and d not in exist and d not in exclusion_disease:
                                    label[k]['d'].append(d)
                                    exist.add(d)
                    elif 'procedure' in i:
                        for j in replace_proc_list:
                            i = i.replace(j, ',')
                        i = re.sub(r'[a-zA-Z]*:', ',', i)
                        i = re.sub(r'[0-9]+ *[\.)]', ',', i)
                        for d in i.split(','):
                            if not d:
                                continue
                            d = re.sub('[\';:,.)(]]|[0-9]+/[0-9]+/[0-9]+', '', d)
                            d = d.strip()
                            if d and d not in exclusion_procedure:
                                label[k]['p'].append(d)

            # json_object = json.dumps(label, indent=2)
            # with open('tmp.json', "w") as outfile:
            #     outfile.write(json_object)
        return df, label

    def gen_synthesis(self, df, id2diseases, template, template_type, dict_types):
        path_to_condition_list = '../../data/TRECCT2021/condition_info/filtered_con2umls.json'
        with open(path_to_condition_list, 'r') as j:
            cond2umls = json.loads(j.read())
        alldisease = list(cond2umls.keys())
        en_stopwords = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        out = []
        tot = 0
        for i in tqdm(range(len(df))):
            exists = set()
            all_disease_token = set()
            pid, text = df.iloc[i]
            pid = str(pid)

            if pid not in id2diseases:
                print(f'No pid {pid}')
                continue
            for dict_type in dict_types:
                if not id2diseases[pid][dict_type]:
                    print(f'No disease pid {pid}, {dict_type}')
                    continue
                out_tmp = {'id': pid, 'pos': [], 'neg': [], 'ret_pos': [], 'ret_neg': []}
                # positive
                for d in id2diseases[pid][dict_type]:
                    exists.add(d.strip())
                    rand = random.randint(1, len(template[template_type[dict_type]]) - 1)
                    out_tmp['pos'].append(template[template_type[dict_type]][rand].replace('[mask]', d))
                    for dt in word_tokenize(d):
                        if dt not in en_stopwords and dt not in '+/:][()':
                            all_disease_token.add(stemmer.stem(dt))

                # negative
                tried = set()
                while len(out_tmp['neg']) < min(len(out_tmp['pos']) * 8, 100) and len(tried) != len(alldisease):
                    d = random.choice(alldisease)
                    tried.add(d)
                    flag = True
                    for dt in word_tokenize(d):
                        if dt not in en_stopwords and stemmer.stem(dt) in all_disease_token:
                            flag = False
                            break
                    if flag:
                        rand = random.randint(1, len(template[template_type[dict_type]]) - 1)
                        out_tmp['neg'].append(
                            template[template_type[dict_type]][rand].replace('[mask]', d.lower().strip()))
                        exists.add(d.strip())
                tot += len(out_tmp['pos']) * 2
                if len(out) > 0 and out[-1]['id'] == out_tmp['id']:
                    for qq in ['pos', 'neg']:
                        out[-1][qq] += out_tmp[qq]
                else:
                    out.append(out_tmp)
            # retrieve
            if len(out) > 0:
                out_tmp['ret_pos'] = self.simple_search(out[-1]['pos'])
                out_tmp['ret_neg'] = self.simple_search(out[-1]['neg'])
        print('total samples', tot)
        return out


class i2b2_doc_2008(i2b2_doc_2009):
    def __init__(self, year):
        indexref = '../../data/TRECCT2021/pyterrier_criteria'
        self.index = pt.IndexFactory.of(indexref)
        self.generate_samples(year)

    def generate_samples(self, year):
        out_path_dia = f'output/i2b2/{year}_dia_8neg.json'
        df1, icd9 = self.read_file(year)
        self.read_judgement(icd9, year)
        template = self.read_template()
        synth_data = self.gen_synthesis(df1, icd9, template, {'d': 'disease'}, ['d'])
        self.write_out(synth_data, out_path_dia)

    def read_judgement(self, id2disease, year):
        path_to_judgement = f'../ehr_section_prediction/i2b2/synth/{year}_docJudgement.json'
        with open(path_to_judgement, 'r') as j:
            alljudge = json.loads(j.read())
        for i in id2disease:
            if i in alljudge:
                for t in alljudge[i]:
                    for d in alljudge[i][t]:
                        if d.strip().lower() not in id2disease[i]['d']:
                            id2disease[i]['d'].append(d.strip().lower())


class i2b2_doc_2010(i2b2_doc_2008):
    def __init__(self, year):
        indexref = '../../data/TRECCT2021/pyterrier_criteria'
        self.index = pt.IndexFactory.of(indexref)
        self.generate_samples(year)

    def generate_samples(self, year):
        out_path_dia = f'output/i2b2/{year}_dia_8neg.json'
        df1, icd9 = self.read_file(year)
        self.read_judgement(icd9, year)
        template = self.read_template()
        synth_data = self.gen_synthesis(df1, icd9, template, {'d': 'disease', 'pd': 'patienthistory'}, ['d', 'pd'])
        self.write_out(synth_data, out_path_dia)

    def read_judgement(self, id2disease, year):
        exclude_list = ['Allergies', 'Social history', 'Family history']
        path_to_judgement = f'../ehr_section_prediction/i2b2/synth/{year}_parsed.csv'
        df = pd.read_csv(path_to_judgement)
        df = df.dropna(how='any', subset=['labels'], axis=0)
        for idx in range(len(df)):
            id, header, _, labels = df.iloc[idx]
            if id not in id2disease:
                id2disease[id] = {'d': [], 'p': [], 'pd': []}
            for dtype in ['d', 'p', 'pd']:
                if dtype not in id2disease[id]:
                    id2disease[id][dtype] = []
            for label in labels.split('|'):
                tmp_list = label[1:-1].split(',')
                if len(tmp_list) == 3:
                    name, name_type, is_pos = ' '.join(tmp_list[:-2]).strip(), tmp_list[-2].strip(), tmp_list[
                        -1].strip()
                else:
                    name, name_type, is_pos = ' '.join(tmp_list[:-2]).strip(), tmp_list[-2].strip(), 'POS'
                if name_type.lower() == 'problem' and is_pos == 'POS':
                    if header in exclude_list:
                        continue
                    elif header == 'Medical history':
                        if name.strip() not in id2disease[id]['pd']:
                            id2disease[id]['pd'].append(name.strip())
                    else:
                        if name.strip() not in id2disease[id]['d']:
                            id2disease[id]['d'].append(name.strip())
                elif name_type.lower() == 'treatment' and is_pos == 'POS':
                    if name.strip() not in id2disease[id]['p']:
                        id2disease[id]['p'].append(name.strip())


class i2b2_doc_2014(i2b2_doc_2008):
    def __init__(self, year):
        indexref = '../../data/TRECCT2021/pyterrier_criteria'
        self.index = pt.IndexFactory.of(indexref)
        self.generate_samples(year)

    def generate_samples(self, year):
        out_path_dia = f'output/i2b2/{year}_dia_8neg.json'
        out_path_pro = f'output/i2b2/{year}_pro_8neg.json'
        df1, icd9 = self.read_file(year)
        self.read_judgement(icd9, year)
        template = self.read_template()
        synth_data = self.gen_synthesis(df1, icd9, template, {'d': 'disease', 'pd': 'patienthistory'}, ['d', 'pd'])
        self.write_out(synth_data, out_path_dia)
        synth_data = self.gen_synthesis(df1, icd9, template, {'p': 'admindrug'}, ['p'])
        self.write_out(synth_data, out_path_pro)

    def read_judgement(self, id2disease, year):
        exclude_list = ['Allergies', 'Social history', 'Family history']
        path_to_judgement = f'../ehr_section_prediction/i2b2/synth/{year}_parsed.csv'
        df = pd.read_csv(path_to_judgement)
        df = df.dropna(how='any', subset=['labels'], axis=0)
        label_diseases = ['DIABETES', 'HYPERTENSION', 'CAD', 'OBESE']
        label_diseases = [i.lower() for i in label_diseases]
        for idx in range(len(df)):
            id, header, _, labels = df.iloc[idx]
            if not labels.strip():
                continue
            if id not in id2disease:
                id2disease[id] = {'d': [], 'p': [], 'pd': []}
            for dtype in ['d', 'p', 'pd']:
                if dtype not in id2disease[id]:
                    id2disease[id][dtype] = []
            for label in labels.split('|'):
                tmp_list = label[1:-1].split(',')
                name, name_type, type1 = ' '.join(tmp_list[:-2]).strip(), tmp_list[-2].strip(), tmp_list[-1].strip()
                name_type = name_type.lower().strip()
                type1 = type1.lower().strip()
                if name_type.strip() in label_diseases:
                    if header in exclude_list:
                        continue
                    elif header == 'Medical history':
                        if name_type not in id2disease[id]['pd']:
                            id2disease[id]['pd'].append(name_type.strip())
                    else:
                        if name_type not in id2disease[id]['d']:
                            id2disease[id]['d'].append(name_type.strip())
                elif name_type == 'MEDICATION'.lower():
                    if name.strip() not in id2disease[id]['p']:
                        id2disease[id]['p'].append(name.strip() + ' (' + type1 + ')')


class CreateTripple():
    def __init__(self, dtype, tokenizer):
        self.path_out = f'output/mimic/tripple_mimic_{dtype}.tsv'
        self.tokenizer = tokenizer
        self.run_all(dtype)

    def run_all(self, dtype):
        df, jfile = self.read_file()
        self.gen_tripple(df, jfile, dtype)

    def tokenize(self, text, pass_p, pass_n):
        token_q = self.tokenizer.tokenize(text)
        token_p = self.tokenizer.tokenize(pass_p)
        token_n = self.tokenizer.tokenize(pass_n)
        max_len_pass = max(len(token_p), len(token_n))
        if len(token_q) + max_len_pass > 500:
            new_token = token_q[:(500-len(token_p))]
            new_text = ''.join(new_token).replace('▁', ' ')
            return new_text.strip()
        else:
            return text.strip()

    def gen_tripple(self, df, jfile, dtype):
        print(self.path_out)
        ddtype = ['pos', 'neg'] if dtype != 'ret' else ['ret_pos', 'ret_neg']
        with open(self.path_out, 'w') as f:
            for i in tqdm(range(len(df))):
                qid = str(df.iloc[i, 0])
                if qid not in jfile:
                    continue
                text = df.iloc[i, 1]
                text = text.lower()
                text = re.sub(r'\[([^\]]+)]', 'NA', text)
                text = re.sub(r'\r|\n|\t|\s\s+', ' ', text)
                for j in range(len(jfile[qid][ddtype[0]])):
                    if j >= len(jfile[qid][ddtype[0]]) or j >= len(jfile[qid][ddtype[1]]):
                        continue
                    pos_text = re.sub(r'\r|\n|\t|\s\s+', ' ', jfile[qid][ddtype[0]][j])
                    neg_text = re.sub(r'\r|\n|\t|\s\s+', ' ', jfile[qid][ddtype[1]][j])
                    text = self.tokenize(text, pos_text, neg_text)
                    f.write("{}\t{}\t{}\n".format(text, pos_text, neg_text))
                f.flush()

    def read_file(self):
        path_to_json = './output/mimic/mimic_dia.json'
        with open(path_to_json, 'r') as f:
            contents = json.loads(f.read())
        for type in ['train', 'test', 'val']:
            path_to_file = f'../ehr_section_prediction/mimicIII/DIA_PLUS_{type}.csv'
            if type == 'train':
                df = pd.read_csv(path_to_file)
            else:
                df.append(pd.read_csv(path_to_file))
        out = {}
        for j in contents:
            out[j['id']] = j
        return df, out


class CreateTripple_i2b2(CreateTripple):
    def __init__(self, dtype, year, tokenizer):
        self.year = year
        self.path_out = f'output/i2b2/tripple_i2b2_{self.year}_{dtype}.tsv'
        self.tokenizer = tokenizer
        self.run_all(dtype)

    def read_file(self):
        path_to_json = f'./output/i2b2/{self.year}_dia.json'
        with open(path_to_json, 'r') as f:
            contents = json.loads(f.read())
        out = {}
        for j in contents:
            out[j['id']] = j

        path_to_file = f'../ehr_section_prediction/i2b2/synth/{self.year}_parsed.csv'
        df = pd.read_csv(path_to_file)
        df = df.dropna(how='any', subset=['text'], axis=0)
        df = df.drop(df[(df['header'] == 'Diagnosis')].index)
        df = df.groupby(['id'])['text'].apply(" ".join).reset_index()
        return df, out


def combine_all_file(dtype):
    path_output = f'./data/tripple/tripple_psu_{dtype}.tsv'
    path_to_dir = 'output'
    os.system('rm {}'.format(path_output))
    filelist = {}
    for path, subdirs, files in os.walk(path_to_dir):
        for name in files:
            if name.split('_')[-1] == f'{dtype}.tsv':
                filelist[name.split('.')[0]] = os.path.join(path, name)
                os.system('cat {} >> {}'.format(os.path.join(path, name), path_output))

def gen_val_db(dtype):
    # dtype = 'ret'
    path_to_tripple = f'./data/tripple/tripple_psu_{dtype}.tsv'
    path_to_tripple_train = f'./data/tripple/tripple_psu_{dtype}_train.tsv'
    path_to_tripple_valid = f'./data/tripple/tripple_psu_{dtype}_valid.tsv'
    with open(path_to_tripple, 'r') as f:
        lines = f.readlines()
    q2lineid = {}
    for idx, line in enumerate(lines):
        q, dp, dn = line.strip().split('\t')
        if q not in q2lineid:
            q2lineid[q] = []
        q2lineid[q].append(line)

    q_list = list(q2lineid.keys())
    trainq = np.random.choice(q_list, int(len(q_list)*0.9), replace=False)
    valq = list(set(q_list).difference(set(trainq)))
    with open(path_to_tripple_train, 'w') as f:
        for q in trainq:
            for l in q2lineid[q]:
                f.write(l)
    with open(path_to_tripple_valid, 'w') as f:
        for q in valq:
            for l in q2lineid[q]:
                f.write(l)

if __name__ == '__main__':
    pt.init(home_dir='/scratch/itee/s4575321/cache/')
    random.seed(123)
    np.random.seed(123)
    # print('mimic iii')
    # Mimiciii()
    # print('2009')
    # i2b2_doc_2009('2009')
    # print('2008')
    # i2b2_doc_2008('2008')
    # print('2006')
    # i2b2_doc_2009('2006')
    #
    # print('2010')
    # i2b2_doc_2010('2010')
    # print('2011')
    # i2b2_doc_2010('2011')
    # print('2012')
    # i2b2_doc_2010('2012')
    #
    # print('2014')
    # i2b2_doc_2014('2014')

    # create tripple
    # tokenizer = AutoTokenizer.from_pretrained('t5-base')
    # for dtype in ['temp', 'ret']:
    #     CreateTripple(dtype, tokenizer)
    #     for i in ['2006', '2008', '2009', '2010', '2011', '2012', '2014']:
    #         CreateTripple_i2b2(dtype, i, tokenizer)
    #     combine_all_file(dtype)
    #     gen_val_db(dtype)
