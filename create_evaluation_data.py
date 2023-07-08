

import json
import pandas as pd
import random

propara_train_tsv_file = 'datasets/propara/grids.v1.train.tsv'
propara_train_json_file = 'datasets/propara/grids.v1.train.json'

distractors_csv_file = 'distractor/distractors.csv'
data_for_evaluation_output_filename = 'data_for_evaluation.csv'


def get_paragraph_titles(filename):
    paragraph_titles = []
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        if "\t\tPROMPT:" not in line:
            continue
        start = line.find("\t\tPROMPT:") + len("\t\tPROMPT:")
        end = line.find("\t-")
        paragraph_title = line[start+1:end]
        paragraph_titles.append(paragraph_title)
    return paragraph_titles


def read_propara_paragraphs(filename):
    f = open(filename, "r")
    lines = f.readlines()
    data = [{} for _ in range(len(lines))]
    for idx, line in enumerate(lines):
        json_object = json.loads(line)
        para_id, texts, participants, states = json_object['para_id'], json_object['sentence_texts'], \
                                               json_object['participants'], json_object['states']
        data[idx]['para_id'], data[idx]['texts'], data[idx]['participants'], data[idx]['states'] = para_id, texts, \
                                                                                                   participants, states
    return data

def create_propara_files(propara_json_file, propara_tsv_file):
    paragraph_titles = get_paragraph_titles(propara_tsv_file)
    d = {}
    for t in paragraph_titles:
        d[t] = 1 if t not in d else d[t] + 1

    data = read_propara_paragraphs(propara_json_file)
    para_id_title_map = {}
    for i in range(len(data)):
        para_id_title_map[data[i]["para_id"]] = paragraph_titles[i]

    converted_data = [{} for _ in range(len(data))]
    for idx, sample in enumerate(data):
        para_id, para_prompt, texts = sample["para_id"], para_id_title_map[sample["para_id"]], sample["texts"]
        converted_data[idx]["para_id"], converted_data[idx]["para_prompt"], converted_data[idx][
            "texts"] = para_id, para_prompt, texts

    return converted_data

def get_data_excluding_source(data, source_paragraph_subject):
    new_data = []
    for d in data:
        if d['para_prompt'] == source_paragraph_subject:
            continue
        new_data.append(d)
    return new_data


def write_output_evaluation_data(data_for_evaluation_output_filename):
    converted_data = create_propara_files(propara_train_json_file, propara_train_tsv_file)
    df = pd.read_csv(distractors_csv_file)
    analogies = df[['sample_id', 'source_subject', 'source_paragraph', 'target_paragraph', 'new_paragraph', 'analogy_type']]
    analogies = analogies.rename(columns={'target_paragraph': 'analogous_target', 'new_paragraph': 'distractor_target'})
    data_for_evaluation = pd.DataFrame(
        columns=['sample_id', 'source_paragraph', 'random1_target_paragraph', 'random2_target_paragraph', 'distractor_target_paragraph',
                 'analogous_target_paragraph', 'analogy_type'])
    random.seed(1)

    for index, row in analogies.iterrows():
        converted_data_excluding_source_subject = get_data_excluding_source(converted_data, row['source_subject'])
        random1, random2 = random.sample(range(len(converted_data_excluding_source_subject)), 2)
        random1_paragraph = '\n'.join(converted_data_excluding_source_subject[random1]['texts'])
        random2_paragraph = '\n'.join(converted_data_excluding_source_subject[random2]['texts'])
        data_for_evaluation = data_for_evaluation.append({  'sample_id' : row['sample_id'],
                                                            'source_paragraph' : row['source_paragraph'],
                                                            'random1_target_paragraph' : random1_paragraph,
                                                            'random2_target_paragraph' : random2_paragraph,
                                                            'distractor_target_paragraph' : row['distractor_target'],
                                                            'analogous_target_paragraph' : row['analogous_target'],
                                                            'analogy_type' : row['analogy_type']}, ignore_index=True)
    data_for_evaluation.to_csv(data_for_evaluation_output_filename)


if __name__ == '__main__':
    write_output_evaluation_data(data_for_evaluation_output_filename)





