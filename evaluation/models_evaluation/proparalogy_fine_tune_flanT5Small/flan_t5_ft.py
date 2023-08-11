import argparse
import json
import os
from collections import defaultdict
import jsonlines
import sklearn
import pandas as pd
import torch
from huggingface_hub import HfFolder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import datasets

PROMPT = open('./binary_prompt.txt').read()


def load_test_data_by_name(name):
    windows_path = f"test_sets\\{name}.jsonl"
    unix_path = f"./test_sets/{name}.jsonl"
    try:
        dict_dataset = load_test_dataset(unix_path)
    except:
        dict_dataset = load_test_dataset(windows_path)
    return dict_dataset

def load_test_dataset(path):
    all_dicts = []
    with open(path) as f:
        for line in f.readlines():
            all_dicts.append(json.loads(line))
    return all_dicts


def preprocess_function(examples, tokenizer):
    inputs = [PROMPT.format(examples['source_paragraph'][i], examples['target_paragraph'][i]) for i in range(len(examples["source_paragraph"]))]
    examples["ground_truth"] = [str(i) for i in examples["ground_truth"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["ground_truth"], max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_test_function(dataset, tokenizer):
    inputs = list(map(lambda instance: PROMPT.format(instance['source_paragraph'], instance['target_paragraph']), dataset))
    ground_truth = list(map(lambda instance: str(instance['ground_truth']), dataset))
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=ground_truth, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_model_save_path(model_name):
    model_save_dir = args.model_save_dir
    model_name_new = model_name.split('/')[1]
    print(model_name_new)
    model_save_path = f"{model_save_dir}{model_name_new}"
    return model_save_path


class Trainer:
    def __init__(self, model_name, save_model_path):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, label_pad_token_id=-100, pad_to_multiple_of=8)
        self.save_model_path = save_model_path
        self.init_train_args()

    def init_train_args(self):
      self.training_args = Seq2SeqTrainingArguments(
            per_device_train_batch_size=16,
            output_dir=f"{self.model_name}_training",
            per_device_eval_batch_size=16,
            fp16=False,  # Overflows with fp16
            learning_rate=1e-5,
            num_train_epochs=args.train_epochs,
            # logging & evaluation strategies
            logging_steps=10,
            evaluation_strategy="no",
            # metric_for_best_model="overall_f1",
            # push to hub parameters
            report_to="tensorboard",
            push_to_hub=False,
            hub_strategy="every_save",
            hub_token=HfFolder.get_token(),)


    def train(self, tokenized_dataset):
        trainer = Seq2SeqTrainer(model=self.model, args=self.training_args, data_collator=self.data_collator, train_dataset=tokenized_dataset)
        trainer.train()
        trainer.model.save_pretrained(self.save_model_path)


class CreateLoadDataset:
    def __init__(self, file_path, save_path):
        self.file_path = file_path
        self.save_path = save_path
        self.objects = self.read_objects_from_jsonlines()

    def save_or_load_dataset(self):
        if os.path.exists(self.save_path):
            return datasets.load_from_disk(self.save_path)
        else:
            dataset = self.turn_to_dataset()
            self.save_dataset(dataset)
            return dataset

    def save_dataset(self, dataset):
        dataset.save_to_disk(self.save_path)

    def read_objects_from_jsonlines(self):
        with jsonlines.open(self.file_path) as reader:
            return [o for o in reader]

    def turn_to_dataset(self):
        return datasets.Dataset.from_dict(self.turn_to_single_dict())

    def turn_to_single_dict(self):
        new_dict = {key: [] for key in self.objects[0].keys()}
        for obj in self.objects:
            for key in new_dict:
                try:
                    new_dict[key].append(obj[key])
                except:
                    print(key)
                    print(obj)
        return new_dict

    def gen(self):
        for object in self.objects:
            yield object


def train():
    # tokenize training set
    create_loader_dataset = CreateLoadDataset(args.raw_data_path, args.dataset_save_dir)
    dataset = create_loader_dataset.save_or_load_dataset()
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(lambda sample: preprocess_function(sample, tokenizer), batched=True)

    # train  model
    model_trainer = Trainer(model_name, get_model_save_path(model_name))
    model_trainer.train(tokenized_dataset)
    print('Finished training')


def write_results_to_csv(sample_ids, type, predictions, ground_truth):
    data = {'sample_id': sample_ids, 'type': type, 'predictions': predictions, 'ground_truth': ground_truth}
    df = pd.DataFrame(data)
    df.to_csv('fine_tuning_results.csv', index=False)


def inference():
    # tokenize test set
    test_dataset = load_test_data_by_name("gold_test_-1_samples_test")
    ground_truth = list(map(lambda instance: int(instance['ground_truth']), test_dataset))
    type =  list(map(lambda instance: 'analogy' if instance['type'] in ['close analogy', 'far analogy'] else instance['type'], test_dataset))
    sample_ids = list(map(lambda instance: instance['sample_id'], test_dataset))



    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_test_set = preprocess_test_function(test_dataset, tokenizer).data['input_ids']

    # Load model
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(get_model_save_path(model_name))
    if torch.cuda.is_available():
        loaded_model.to(torch.device("cuda"))

    # Predict on test set
    predictions = []
    for instance in tokenized_test_set:
        inputs = torch.tensor([instance]).to(torch.device("cuda"))
        outputs = loaded_model.generate(inputs, max_new_tokens=400, do_sample=False)
        predictions.append(int(''.join([tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(outputs.shape[0])])))

    print('Sample ids:')
    print(sample_ids)
    print("Length of sample ids: " + str(len(sample_ids)))
    print('Predictions:')
    print(predictions)
    print("Length of predictions: " + str(len(predictions)))
    print('Ground Truth:')
    print(ground_truth)
    print("Length of ground truth: " + str(len(ground_truth)))
    print('Type:')
    print(type)
    print("Length of type:" + str(len(type)))
    write_results_to_csv(sample_ids, type, predictions, ground_truth)


    # Group predictions and ground truths by type
    results_by_type = defaultdict(list)
    for prediction, true_value, type in zip(predictions, ground_truth, type):
        results_by_type[type].append((prediction, true_value))

    # Compute accuracy for each type
    accuracy_by_type = {}
    for type, results in results_by_type.items():
        predictions_per_type, true_values = zip(*results)  # unzip the list of pairs
        accuracy = sklearn.metrics.accuracy_score(true_values, predictions_per_type)
        accuracy_by_type[type] = accuracy

    # Print accuracy per type
    for type, accuracy in accuracy_by_type.items():
        print(f'Accuracy for type {type}: {accuracy}')

    print('Overall Accuracy: ', sklearn.metrics.accuracy_score(ground_truth, predictions))
    print('Overall F1 Score: ', sklearn.metrics.f1_score(ground_truth, predictions))


   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", dest="raw_data_path", type=str, default="./train_samples/annotated_data.jsonl")
    parser.add_argument("--dataset_save_dir", dest="dataset_save_dir", type=str, default="./datasets/annotated_dataset.hf")
    parser.add_argument("--model_save_dir", dest="model_save_dir", type=str, default="./models/")
    parser.add_argument("--inference_test_set", dest="inference_test_set", type=str, default="gold_test_-1_samples_test")
    parser.add_argument("--pretrained_model_path", dest="pretrained_model_path", type=str, default="./models/flan-t5-small")
    parser.add_argument("--model_name", dest="model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--test_save_dir", dest="test_save_dir", type=str, default="./test_sets")
    parser.add_argument("--run_train", dest="run_train", action="store_true")
    parser.add_argument("--run_inference", dest="run_inference", action="store_true")
    parser.add_argument("--inference_batch_size", dest="inference_batch_size", type=int, default=32)
    parser.add_argument("--train_epochs", dest="train_epochs", type=int, default=1)
    parser.add_argument("--inference_size", dest="inference_size", type=int, default=-1)  # -1 is run on all data
    args = parser.parse_args()

    if args.run_train:
        train()

    if args.run_inference:
        inference()
