
import json
import argparse
import openai
import time
import csv

propara_train_tsv_file = 'datasets/propara/grids.v1.train.tsv'
propara_train_json_file = 'datasets/propara/grids.v1.train.json'


# the output analogies candidates. The first is all data (4680 candidates), the second is the filtered (4288 candidates)
far_analogies_propara_topics_relations_mappings_csv = 'pipeline/analogy_candidates_generation/analogy_candidates_dataset/GPT3_analogies_propara_topics_relations_mappings.csv'
far_analogies_propara_topics_relations_filtered_csv = 'pipeline/analogy_candidates_generation/analogy_candidates_dataset/GPT3_analogies_propara_topics_relations_filtered.csv'

# the two separate prompts
examples_subject_relations_generate_paragraph_txt = 'pipeline/analogy_candidates_generation/prompts/GPT3_generate_paragraphs.txt'
examples_propara_topics_relations_mappings_txt = 'pipeline/analogy_candidates_generation/prompts/GPT3_find_analogy_relations.txt'

domains = ['Engineering', 'Natural Sciences', 'Social Sciences', 'Biomedical and Health Sciences']
num_trials_per_domain = 3

# insert your OPENAI key here
OPENAI_API_KEY = ''
openai.api_key = OPENAI_API_KEY


def call_gpt3(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if response:
        if response.choices and response.choices[0]:
            res = response.choices[0].text.strip()
            return res


def gpt3_generate_target_propara_topics_analogies(start_paragraph_id, end_paragraph_id):
    converted_data = create_propara_files()
    # this is the header, you should write it only in the first run (every run appends to the csv, so you do not want to append the header again)
    # header = ['source_id', 'source_subject', 'source_domain', 'target_domain', 'target_subject', 'target_field', 'relations', 'source_paragraph', 'target_paragraph']

    with open(far_analogies_propara_topics_relations_mappings_csv, 'r') as file:
        csvreader = csv.reader(file)
        processed_ids = set()
        for row in csvreader:
            processed_ids.add(row[0])

    with open(examples_propara_topics_relations_mappings_txt) as f:
        examples_topics_relations_prompt = "".join(f.readlines())
    with open(examples_subject_relations_generate_paragraph_txt) as f:
        examples_subject_relations_generate_paragraph_prompt = "".join(f.readlines())

    for paragraph in converted_data[start_paragraph_id:end_paragraph_id]:
        paragraph_prompt, paragraph_id, paragraph_text = paragraph["para_prompt"], paragraph["para_id"], paragraph["texts"]
        if paragraph_id in processed_ids:
            print("Already run on paragraph: " + paragraph_id)
            continue

        # prompt for outputing the field of the target domain
        source_domain_prompt = "Write the field of the following subject: " + '"' + paragraph_prompt + '"' + " Write FIELD: and then one of the following options: "
        for target_domain in domains:
            source_domain_prompt = source_domain_prompt + " " + target_domain + " or"
        source_domain = call_gpt3(source_domain_prompt[:-2])
        _, source_domain = source_domain.split('FIELD:')
        source_domain = source_domain.strip()
        time.sleep(1)
        data = []
        seen_target_paragraph_prompts = set()
        for target_domain in domains:
            count = 0
            while count < num_trials_per_domain:
                # concat for creating the prompt of finding analogous subject
                examples_topics_relations_prompt = examples_topics_relations_prompt + "\n\n" + "Inputs:" + "\n" + "BASE:" + \
                                                       paragraph_prompt + "\n" + "TARGET_DOMAIN: One of the fields of " \
                                                       + target_domain + "\n" + "Outputs:\n"
                result = call_gpt3(examples_topics_relations_prompt)
                time.sleep(1)

                # parse the results from GPT, and validate
                if len(result.split('TARGET_FIELD:')) != 2:
                    continue
                target_subj, after = result.split('TARGET_FIELD:')
                if len(target_subj.split('TARGET:')) != 2:
                    continue
                before, target_subj = target_subj.split('TARGET:')
                target_subj = target_subj[:-1]
                if target_subj in seen_target_paragraph_prompts:
                    continue
                seen_target_paragraph_prompts.add(target_subj)
                target_field, after = after.split('SIMILAR_RELATIONS:')
                target_field = target_field[1:-1]
                common_relations = after[1:].strip()

                # concat for creating the full prompt for writing a paragraph
                examples_subject_relations_generate_source_paragraph_prompt = examples_subject_relations_generate_paragraph_prompt + "\n\n" + "Inputs:" + "\n" + "SUBJECT:" + \
                    paragraph_prompt + "\n" + "RELATIONS:" + "\n" + extract_relations(common_relations, 'source') + "\n" + "PARAGRAPH:\n"
                examples_subject_relations_generate_target_paragraph_prompt = examples_subject_relations_generate_paragraph_prompt + "\n\n" + "Inputs:" + "\n" + "SUBJECT:" + \
                                                                              target_subj + "\n" + "RELATIONS:" + "\n" + \
                                                                              extract_relations(common_relations, 'target') + "\n" + "PARAGRAPH:\n"

                # generate both the source and target paragraphs.
                source_paragraph = call_gpt3(examples_subject_relations_generate_source_paragraph_prompt)
                time.sleep(1)
                target_paragraph = call_gpt3(examples_subject_relations_generate_target_paragraph_prompt)
                time.sleep(1)
                data.append([paragraph_id, paragraph_prompt, source_domain, target_domain, target_subj, target_field, common_relations, source_paragraph, target_paragraph])
                count += 1

        if len(data) > 0:
            # append new generated analogy candidates
            with open(far_analogies_propara_topics_relations_mappings_csv, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)


def extract_relations(relations, side='source'):
    i = 0 if side == 'source' else 1
    res = ""
    for relation in relations.splitlines():
        res += relation.split('like')[i].strip() + "\n"
    return res[:-1]


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


def filter_partial_relations(data, output_file):
    output_rows = []
    with open(data, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        output_rows.append(header)
        for row in csvreader:
            relations = row[6]
            num_relations = relations.split().count('like')
            if num_relations < 3:
                continue
            output_rows.append(row)
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)



def main():
    print(1)
    # gpt3_generate_target_propara_topics_analogies(0, 50)
    # filter_partial_relations(far_analogies_propara_topics_relations_mappings_csv, far_analogies_propara_topics_relations_filtered_csv)



if __name__ == '__main__':
    main()