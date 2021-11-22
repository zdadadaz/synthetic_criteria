import pandas as pd
import os
import string
from functools import reduce
import csv
import re

class Mimi3:
    def __init__(self):
        path_to_dir = '/scratch/itee/s4575321/data/mimicIII/full'
        save_dir = 'mimicIII/'
        self.run_all_process(path_to_dir, save_dir)
        self.run_all_diagnosis(path_to_dir, save_dir)

    def run_all_process(self, mimic_dir, save_dir, seed=123):
        # load dataframes
        task_name = "PRO_PLUS"
        mimic_dia_names = pd.read_csv(os.path.join(mimic_dir, "D_ICD_PROCEDURES.csv"), dtype={"ICD9_CODE": str})
        mimic_diagnoses = pd.read_csv(os.path.join(mimic_dir, "PROCEDURES_ICD.csv"), dtype={"ICD9_CODE": str})

        mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
        mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

        # filter notes
        mimic_notes = self.filter_notes(mimic_notes, mimic_admissions, admission_text_only=True)

        # only keep relevant columns
        mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]

        # drop all rows without diagnoses codes
        mimic_diagnoses = mimic_diagnoses.dropna(how='any', subset=['ICD9_CODE', 'HADM_ID'], axis=0)

        # CREATE LABELS FOR DIAGNOSES NAMES

        # remove punctuation and split words of diagnoses descriptions
        mimic_dia_names["DIA_NAMES"] = mimic_dia_names.LONG_TITLE.str.replace('[{}]'.format(string.punctuation), '') \
            .str.lower().str.split()

        mimic_dia_names["DIA_NAMES"] = mimic_dia_names.DIA_NAMES.apply(
            lambda x: " ".join([word for word in x]))

        # mimic_dia_names["DIA_NAMES"] = mimic_dia_names.DIA_NAMES.apply(
        #     lambda x: " ".join(set([word for word in x if word not in list(stopwords.words('english'))])))

        # CREATE LABELS FOR 3 DIGIT CODES

        # Truncate codes to 3 digits
        mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)

        mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.SHORT_CODE.apply(lambda x: x[:3])

        # CREATE LABELS FOR 4 DIGIT CODES

        # Truncate codes to 4 digits
        mimic_diagnoses["LONG_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)

        mimic_diagnoses["LONG_CODE"] = mimic_diagnoses.LONG_CODE.apply(lambda x: x[:4])

        # MERGE DESCRIPTION WITH ADMISSION CODES
        admissions_with_dia_names = pd.merge(mimic_diagnoses, mimic_dia_names[["ICD9_CODE", "DIA_NAMES"]],on="ICD9_CODE", how="left")
        admissions_with_dia_names["DIA_NAMES"] = admissions_with_dia_names.DIA_NAMES.fillna("")

        # short
        mimic_dia_names_short = mimic_dia_names.rename(columns={'ICD9_CODE': 'SHORT_CODE'})
        admissions_with_dia_names_short = pd.merge(mimic_diagnoses, mimic_dia_names_short[["SHORT_CODE", "DIA_NAMES"]],
                                                   on="SHORT_CODE", how="left")
        admissions_with_dia_names_short["DIA_NAMES"] = admissions_with_dia_names_short.DIA_NAMES.fillna("")
        admissions_with_dia_names_short["DIA_NAMES_SHORT"] = admissions_with_dia_names_short["DIA_NAMES"]
        admissions_with_dia_names_short = admissions_with_dia_names_short.drop(columns=['DIA_NAMES'], axis=1)

        # long
        mimic_dia_names_long = mimic_dia_names.rename(columns={'ICD9_CODE': 'LONG_CODE'})
        admissions_with_dia_names_long = pd.merge(mimic_diagnoses, mimic_dia_names_long[["LONG_CODE", "DIA_NAMES"]],
                                                  on="LONG_CODE", how="left")
        admissions_with_dia_names_long["DIA_NAMES"] = admissions_with_dia_names_long.DIA_NAMES.fillna("")
        admissions_with_dia_names_long["DIA_NAMES_LONG"] = admissions_with_dia_names_long["DIA_NAMES"]
        admissions_with_dia_names_long = admissions_with_dia_names_long.drop(columns=['DIA_NAMES'], axis=1)

        # GROUP CODES BY ADMISSION
        code_short = admissions_with_dia_names.groupby(['HADM_ID'])['SHORT_CODE'].apply("|".join).reset_index()
        code_long = admissions_with_dia_names.groupby(['HADM_ID'])['LONG_CODE'].apply("|".join).reset_index()

        grouped_dia_names = admissions_with_dia_names.groupby(['HADM_ID'])['DIA_NAMES'].apply("|".join).reset_index()
        grouped_dia_names_short = admissions_with_dia_names_short.groupby(['HADM_ID'])['DIA_NAMES_SHORT'].apply(
            "|".join).reset_index()
        grouped_dia_names_long = admissions_with_dia_names_long.groupby(['HADM_ID'])['DIA_NAMES_LONG'].apply(
            "|".join).reset_index()

        # COMBINE 3-DIGIT CODES, 4-DIGIT CODES AND DIAGNOSES NAMES

        # combine into one dataframe
        combined_df = reduce(lambda left, right: pd.merge(left, right, on=['HADM_ID'], how='outer'),
                             [grouped_dia_names, grouped_dia_names_short, grouped_dia_names_long, code_short,
                              code_long])

        combined_df['DIA_NAMES'] = combined_df.DIA_NAMES.apply(
            lambda x: '' if len(x.replace('||', '')) == 1 else re.sub(r'\|+', '|',x))
        combined_df['DIA_NAMES_SHORT'] = combined_df.DIA_NAMES_SHORT.apply(
            lambda x: '' if len(x.replace('||', '')) == 1 else re.sub(r'\|+', '|',x))
        combined_df['DIA_NAMES_LONG'] = combined_df.DIA_NAMES_LONG.apply(
            lambda x: '' if len(x.replace('||', '')) == 1 else re.sub(r'\|+', '|',x))
        combined_df['SHORT_CODE'] = combined_df.SHORT_CODE.apply(
            lambda x: '' if len(x.replace('||', '')) == 1 else re.sub(r'\|+', '|',x))
        combined_df['LONG_CODE'] = combined_df.LONG_CODE.apply(
            lambda x: '' if len(x.replace('||', '')) == 1 else re.sub(r'\|+', '|',x))

        # combine into one column
        # combined_df["LABELS"] = combined_df["DIA_NAMES"] + "|.|" + combined_df["DIA_NAMES_SHORT"] + "|.|" + combined_df["DIA_NAMES_LONG"]

        # remove duplicates, sort and join with comma
        # combined_df["LABELS"] = combined_df.LABELS.str.split(" ").apply(lambda x: ",".join(sorted(set(x))))

        # merge discharge summaries into diagnoses table
        notes_diagnoses_df = pd.merge(
            combined_df[['HADM_ID', 'DIA_NAMES', 'SHORT_CODE', 'DIA_NAMES_SHORT', 'LONG_CODE', 'DIA_NAMES_LONG']],
            mimic_notes, how='inner', on='HADM_ID')

        self.save_mimic_split_patient_wise(notes_diagnoses_df,
                                           save_dir=save_dir,
                                           task_name=task_name,
                                           seed=seed,
                                           isDiag=False)

    def run_all_diagnosis(self, mimic_dir, save_dir, seed=123):
        # load dataframes
        task_name = "DIA_PLUS"
        mimic_dia_names = pd.read_csv(os.path.join(mimic_dir, "D_ICD_DIAGNOSES.csv"), dtype={"ICD9_CODE": str})
        mimic_diagnoses = pd.read_csv(os.path.join(mimic_dir, "DIAGNOSES_ICD.csv"))

        mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
        mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

        # filter notes
        mimic_notes = self.filter_notes(mimic_notes, mimic_admissions, admission_text_only=True)

        # only keep relevant columns
        mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]

        # drop all rows without diagnoses codes
        mimic_diagnoses = mimic_diagnoses.dropna(how='any', subset=['ICD9_CODE', 'HADM_ID'], axis=0)

        # CREATE LABELS FOR DIAGNOSES NAMES

        # remove punctuation and split words of diagnoses descriptions
        mimic_dia_names["DIA_NAMES"] = mimic_dia_names.LONG_TITLE.str.replace('[{}]'.format(string.punctuation), '') \
            .str.lower().str.split()

        mimic_dia_names["DIA_NAMES"] = mimic_dia_names.DIA_NAMES.apply(
            lambda x: " ".join([word for word in x]))

        # mimic_dia_names["DIA_NAMES"] = mimic_dia_names.DIA_NAMES.apply(
        #     lambda x: " ".join(set([word for word in x if word not in list(stopwords.words('english'))])))

        # CREATE LABELS FOR 3 DIGIT CODES

        # Truncate codes to 3 digits (preserve 3 digits for E and V codes aswell)
        mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)
        mimic_diagnoses.loc[
            mimic_diagnoses['SHORT_CODE'].str.startswith("V"), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
            lambda x: x[:4])
        mimic_diagnoses.loc[
            mimic_diagnoses['SHORT_CODE'].str.startswith("E"), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
            lambda x: x[:4])
        mimic_diagnoses.loc[(~mimic_diagnoses.SHORT_CODE.str.startswith("E")) & (
            ~mimic_diagnoses.SHORT_CODE.str.startswith("V")), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
            lambda x: x[:3])

        # CREATE LABELS FOR 4 DIGIT CODES

        # Truncate codes to 4 digits (preserve 4 digits for E and V codes aswell)
        mimic_diagnoses["LONG_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)
        mimic_diagnoses.loc[
            mimic_diagnoses['LONG_CODE'].str.startswith("V"), 'LONG_CODE'] = mimic_diagnoses.LONG_CODE.apply(
            lambda x: x[:5])
        mimic_diagnoses.loc[
            mimic_diagnoses['LONG_CODE'].str.startswith("E"), 'LONG_CODE'] = mimic_diagnoses.LONG_CODE.apply(
            lambda x: x[:5])
        mimic_diagnoses.loc[(~mimic_diagnoses.LONG_CODE.str.startswith("E")) & (
            ~mimic_diagnoses.LONG_CODE.str.startswith("V")), 'LONG_CODE'] = mimic_diagnoses.LONG_CODE.apply(
            lambda x: x[:4])

        # MERGE DESCRIPTION WITH ADMISSION CODES
        admissions_with_dia_names = pd.merge(mimic_diagnoses, mimic_dia_names[["ICD9_CODE", "DIA_NAMES"]],
                                             on="ICD9_CODE", how="left")
        admissions_with_dia_names["DIA_NAMES"] = admissions_with_dia_names.DIA_NAMES.fillna("")

        # short
        mimic_dia_names_short = mimic_dia_names.rename(columns={'ICD9_CODE': 'SHORT_CODE'})
        admissions_with_dia_names_short = pd.merge(mimic_diagnoses, mimic_dia_names_short[["SHORT_CODE", "DIA_NAMES"]],
                                                   on="SHORT_CODE", how="left")
        admissions_with_dia_names_short["DIA_NAMES"] = admissions_with_dia_names_short.DIA_NAMES.fillna("")
        admissions_with_dia_names_short["DIA_NAMES_SHORT"] = admissions_with_dia_names_short["DIA_NAMES"]
        admissions_with_dia_names_short = admissions_with_dia_names_short.drop(columns=['DIA_NAMES'], axis=1)

        # long
        mimic_dia_names_long = mimic_dia_names.rename(columns={'ICD9_CODE': 'LONG_CODE'})
        admissions_with_dia_names_long = pd.merge(mimic_diagnoses, mimic_dia_names_long[["LONG_CODE", "DIA_NAMES"]],
                                                  on="LONG_CODE", how="left")
        admissions_with_dia_names_long["DIA_NAMES"] = admissions_with_dia_names_long.DIA_NAMES.fillna("")
        admissions_with_dia_names_long["DIA_NAMES_LONG"] = admissions_with_dia_names_long["DIA_NAMES"]
        admissions_with_dia_names_long = admissions_with_dia_names_long.drop(columns=['DIA_NAMES'], axis=1)

        # GROUP CODES BY ADMISSION
        code_short = admissions_with_dia_names.groupby(['HADM_ID'])['SHORT_CODE'].apply("|".join).reset_index()
        code_long = admissions_with_dia_names.groupby(['HADM_ID'])['LONG_CODE'].apply("|".join).reset_index()

        grouped_dia_names = admissions_with_dia_names.groupby(['HADM_ID'])['DIA_NAMES'].apply("|".join).reset_index()
        grouped_dia_names_short = admissions_with_dia_names_short.groupby(['HADM_ID'])['DIA_NAMES_SHORT'].apply(
            "|".join).reset_index()
        grouped_dia_names_long = admissions_with_dia_names_long.groupby(['HADM_ID'])['DIA_NAMES_LONG'].apply(
            "|".join).reset_index()

        # COMBINE 3-DIGIT CODES, 4-DIGIT CODES AND DIAGNOSES NAMES

        # combine into one dataframe
        combined_df = reduce(lambda left, right: pd.merge(left, right, on=['HADM_ID'], how='outer'),
                             [grouped_dia_names, grouped_dia_names_short, grouped_dia_names_long, code_short, code_long])

        combined_df['DIA_NAMES'] = combined_df.DIA_NAMES.apply(lambda x: '' if len(x.replace('||', ''))==1 else re.sub(r'\|+', '|',x))
        combined_df['DIA_NAMES_SHORT'] = combined_df.DIA_NAMES_SHORT.apply(lambda x: '' if len(x.replace('||', ''))==1 else re.sub(r'\|+', '|',x))
        combined_df['DIA_NAMES_LONG'] = combined_df.DIA_NAMES_LONG.apply(lambda x: '' if len(x.replace('||', ''))==1 else re.sub(r'\|+', '|',x))
        combined_df['SHORT_CODE'] = combined_df.SHORT_CODE.apply(lambda x: '' if len(x.replace('||', '')) == 1 else re.sub(r'\|+', '|',x))
        combined_df['LONG_CODE'] = combined_df.LONG_CODE.apply(lambda x: '' if len(x.replace('||', '')) == 1 else re.sub(r'\|+', '|',x))

        # combine into one column
        # combined_df["LABELS"] = combined_df["DIA_NAMES"] + "|.|" + combined_df["DIA_NAMES_SHORT"] + "|.|" + combined_df["DIA_NAMES_LONG"]

        # remove duplicates, sort and join with comma
        # combined_df["LABELS"] = combined_df.LABELS.str.split(" ").apply(lambda x: ",".join(sorted(set(x))))

        # merge discharge summaries into diagnoses table
        notes_diagnoses_df = pd.merge(combined_df[['HADM_ID', 'DIA_NAMES', 'SHORT_CODE', 'DIA_NAMES_SHORT', 'LONG_CODE', 'DIA_NAMES_LONG']], mimic_notes, how='inner', on='HADM_ID')

        self.save_mimic_split_patient_wise(notes_diagnoses_df,
                                           save_dir=save_dir,
                                           task_name=task_name,
                                           seed=seed)

    def write_codes_to_file(self, icd_codes, data_path):
        # save ICD codes in an extra file
        with open(os.path.join(data_path, "ALL_DIAGNOSES_PLUS_CODES.txt"), "w", encoding="utf-8") as icd_file:
            icd_file.write(" ".join(icd_codes))

    def filter_notes(self, notes_df: pd.DataFrame, admissions_df: pd.DataFrame,
                     admission_text_only=False) -> pd.DataFrame:
        """
        Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
        their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
        """
        # filter out newborns
        adm_grownups = admissions_df[admissions_df.ADMISSION_TYPE != "NEWBORN"]
        notes_df = notes_df[notes_df.HADM_ID.isin(adm_grownups.HADM_ID)]

        # remove notes with no TEXT or HADM_ID
        notes_df = notes_df.dropna(subset=["TEXT", "HADM_ID"])

        # filter discharge summaries
        notes_df = notes_df[notes_df.CATEGORY == "Discharge summary"]

        # remove duplicates and keep the later ones
        notes_df = notes_df.sort_values(by=["CHARTDATE"])
        notes_df = notes_df.drop_duplicates(subset=["TEXT"], keep="last")

        # combine text of same admissions (those are usually addendums)
        combined_adm_texts = notes_df.groupby('HADM_ID')['TEXT'].apply(lambda x: '\n\n'.join(x)).reset_index()
        notes_df = notes_df[notes_df.DESCRIPTION == "Report"]
        notes_df = notes_df[["HADM_ID", "ROW_ID", "SUBJECT_ID", "CHARTDATE"]]
        notes_df = notes_df.drop_duplicates(subset=["HADM_ID"], keep="last")
        notes_df = pd.merge(combined_adm_texts, notes_df, on="HADM_ID", how="inner")

        # strip texts from leading and trailing and white spaces
        notes_df["TEXT"] = notes_df["TEXT"].str.strip()

        # remove entries without admission id, subject id or text
        notes_df = notes_df.dropna(subset=["HADM_ID", "SUBJECT_ID", "TEXT"])

        if admission_text_only:
            # reduce text to admission-only text
            notes_df = self.filter_admission_text(notes_df)

        return notes_df

    def filter_admission_text(self, notes_df) -> pd.DataFrame:
        """
        Filter text information by section and only keep sections that are known on admission time.
        """
        admission_sections = {
            "CHIEF_COMPLAINT": "chief complaint:",
            "PRESENT_ILLNESS": "present illness:",
            "MEDICAL_HISTORY": "medical history:",
            "MEDICATION_ADM": "medications on admission:",
            "ALLERGIES": "allergies:",
            "PHYSICAL_EXAM": "physical exam:",
            "FAMILY_HISTORY": "family history:",
            "SOCIAL_HISTORY": "social history:"
        }

        # replace linebreak indicators
        notes_df['TEXT'] = notes_df['TEXT'].str.replace(r"\n", r"\\n")

        # extract each section by regex
        for key in admission_sections.keys():
            section = admission_sections[key]
            notes_df[key] = notes_df.TEXT.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'
                                                      .format(section))

            notes_df[key] = notes_df[key].str.replace(r'\\n', r' ')
            notes_df[key] = notes_df[key].str.strip()
            notes_df[key] = notes_df[key].fillna("")
            notes_df[notes_df[key].str.startswith("[]")][key] = ""

        # filter notes with missing main information
        notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                            (notes_df.MEDICAL_HISTORY != "")]

        # add section headers and combine into TEXT_ADMISSION
        notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                        + '\n\n' +
                                        "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                        + '\n\n' +
                                        "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                        + '\n\n' +
                                        "MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                        + '\n\n' +
                                        "ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                        + '\n\n' +
                                        "PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                        + '\n\n' +
                                        "FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                        + '\n\n' +
                                        "SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str))

        return notes_df

    def save_mimic_split_patient_wise(self, df, save_dir, task_name, seed=123, isDiag=True):
        """
        Splits a MIMIC dataframe into 70/10/20 train, val, test with no patient occuring in more than one set.
        Uses ROW_ID as ID column and save to save_path.
        """
        column_list = ["ID", "TEXT", 'DIA_NAMES', 'SHORT_CODE', 'DIA_NAMES_SHORT', 'LONG_CODE', 'DIA_NAMES_LONG']
        if not isDiag:
            column_list_pro = ["ID", "TEXT", 'RPO_NAMES', 'SHORT_CODE', 'PRO_NAMES_SHORT', 'LONG_CODE', 'PRO_NAMES_LONG']

        # Load prebuilt MIMIC patient splits
        data_split = {"train": pd.read_csv("tasks/mimic_train.csv"),
                      "val": pd.read_csv("tasks/mimic_val.csv"),
                      "test": pd.read_csv("tasks/mimic_test.csv")}

        # Use row id as general id and cast to int
        df = df.rename(columns={'HADM_ID': 'ID'})
        df.ID = df.ID.astype(int)

        # Create path to task data
        os.makedirs(save_dir, exist_ok=True)

        # Save splits to data folder
        for split_name in ["train", "val", "test"]:
            split_set = df[df.SUBJECT_ID.isin(data_split[split_name].SUBJECT_ID)].sample(frac=1,
                                                                                         random_state=seed)[column_list]
            # lower case column names
            if isDiag:
                split_set.columns = map(str.lower, column_list)
            else:
                split_set.columns = map(str.lower, column_list_pro)

            split_set.to_csv(os.path.join(save_dir, "{}_{}.csv".format(task_name, split_name)),
                             index=False,
                             quoting=csv.QUOTE_ALL)


if __name__ == '__main__':
    Mimi3()
