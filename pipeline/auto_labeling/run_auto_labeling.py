
# from datetime import datetime
# import tiktoken
# from tqdm import tqdm
# from openai_api import predict_sample_openai_chatgpt

import os
import pandas as pd


LABEL_COL = 'isAnalogy(0-3)'
RATIONALE = 'RATIONALE'
RELATIONS = 'relations'

test_indices = (0, 4286)
test_indices_str = f"{test_indices[0]}_{test_indices[1]}"

train_data_path = 'auto_label_data/auto_label_train.csv'
test_data_path = 'auto_label_data/shuffled_data_for_prediction_2905.csv'
cache_dir = 'auto_label_data/cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
output_path = os.path.join(cache_dir, f'predictions_{test_indices_str}.csv')

input_columns = ['source_subject', 'source_paragraph', 'target_subject', 'target_paragraph', 'relations']
EXTRA_OUTPUT_COLUMNS = ['analogy type', 'not analogy reasons']
output_columns = [LABEL_COL]
print(f"output_path: {output_path}")

instructions = """Your task is to rate how analogous are paragraph pairs from 0 (non-analogous) to 3 (very analogous) based on whether they describe similar underlying processes or mechanisms."""

REQUIRED_NUM_RELATIONS_BRUCKETS = 6
MAX_TOKENS = 4000
cache_path = None  # TODO update CSV path when there is a cache


def main():
    train_df = read_and_process_train()

    already_generated_data, test_df = read_and_process_test()

    full_prompt = build_prompt(train_df)

    predicted_items = []
    for idx, (_, r) in tqdm(enumerate(test_df.iterrows()), total=len(test_df), desc='predicting'):
        if r[RELATIONS].count(")") >= REQUIRED_NUM_RELATIONS_BRUCKETS:
            prompt_input = build_prompt_from_columns(r, input_columns)
            curr_prompt = f"{full_prompt}\n\n{prompt_input}\nLABEL: "
            if idx == 0:
                print(f"PROMPT EXAMPLE")
                print(curr_prompt)
            gpt4_full_pred = get_gpt4_response(r.to_dict(), curr_prompt, max_tokens=MAX_TOKENS)
            if gpt4_full_pred is None:
                dump_cache_and_exit(already_generated_data, idx, predicted_items)
            predicted_label = gpt4_full_pred['prediction'].strip().split("\n")[0].strip().split(" ")[0].strip()
            print(f"idx: {idx},  predicted: {predicted_label}\n")
            predicted_label = int(predicted_label)
        else:
            print(f"Incomplete relations {r[RELATIONS]}")
            predicted_label = 0

        r['predicted_label'] = predicted_label

        predicted_items.append(r)

    predicted_items_df_with_already_generated = dump_predictions(already_generated_data, predicted_items)
    print("*** PREDICTIONS VALUE COUNTS ***")
    print(predicted_items_df_with_already_generated['predicted_label'].value_counts())


def dump_predictions(already_generated_data, predicted_items):
    predicted_items_df = pd.DataFrame(predicted_items)
    if len(already_generated_data) > 0:
        predicted_items_df_with_already_generated = pd.concat([predicted_items_df, already_generated_data])
    else:
        predicted_items_df_with_already_generated = predicted_items_df
    print(
        f"Got df at length {len(predicted_items_df_with_already_generated)}, which are {len(predicted_items_df)} predicted and {len(already_generated_data)} already generated writing to {os.path.abspath(output_path)}")
    predicted_items_df_with_already_generated.to_csv(output_path, index=False)
    return predicted_items_df_with_already_generated


def dump_cache_and_exit(already_generated_data, idx, predicted_items):
    ' Dump all DF to CSV with a path with the current timestamp and the dataframe length and exit '
    print(f"Failed to get response for idx: {idx}")
    print(f"Dumping all DF to CSV with a path with the current timestamp and the dataframe length and exit")
    ''' Add a timestamp and number of items to the output path '''
    predicted_items_df = pd.DataFrame(predicted_items)
    if len(already_generated_data) > 0:
        predicted_items_df_with_already_generated = pd.concat([predicted_items_df, already_generated_data])
    else:
        predicted_items_df_with_already_generated = predicted_items_df
    cache_output_path = output_path.replace(".csv",
                                            f"_predicted_items_{len(predicted_items_df_with_already_generated)}_test_indices_{test_indices_str}_items_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv")
    print(
        f"*** ERROR, dumping of len {len(predicted_items_df_with_already_generated)} to output_path: {cache_output_path}")
    predicted_items_df_with_already_generated.to_csv(cache_output_path, index=False)
    exit()


def build_prompt(train_df):
    full_prompt = f"{instructions}\n\n"
    for _, r in train_df.iterrows():
        prompt_input = build_prompt_from_columns(r, input_columns)
        prompt_output = build_prompt_from_columns(r, output_columns)
        full_prompt += f"{prompt_input}\n{prompt_output}\n\n"
    full_prompt = full_prompt.rstrip('\n')
    num_tokens = num_tokens_from_string(full_prompt, "gpt2")
    print(f"Prompt # Tokens: {num_tokens}")
    return full_prompt


def read_and_process_test():
    test_df_full = pd.read_csv(test_data_path)
    test_df_full['idx'] = list(range(len(test_df_full)))
    test_df = test_df_full[test_indices[0]:test_indices[1]]
    for c in ['source_paragraph', 'target_paragraph']:
        test_df[c] = test_df[c].apply(lambda x: x.replace("\n", ""))
    test_df['relations'] = test_df['relations'].apply(lambda x: x.replace("\n", "."))

    ''' Filter already generated data (cache) '''
    if cache_path and os.path.exists(cache_path):
        already_generated_data = pd.read_csv(cache_path)
        print(f"Got {len(already_generated_data)} already generated samples")
        test_df_filtered = test_df[~test_df['idx'].isin(already_generated_data['idx'])]
        print(f"After filter {len(test_df_filtered)} samples, before: {len(test_df)}")
        test_df = test_df_filtered
    else:
        already_generated_data = pd.DataFrame()
    return already_generated_data, test_df


def read_and_process_train():
    df = pd.read_csv(train_data_path)
    print(f"Read {len(df)} from {train_data_path}")
    df = df[~df['isAnalogy(0-3)'].isna()]
    df['isAnalogy(0-3)'] = df['isAnalogy(0-3)'].apply(lambda x: int(x))
    df['idx'] = list(range(len(df)))
    print(f"Read {len(df)} items")
    print(df.iloc[0])
    for c in ['source_paragraph', 'target_paragraph']:
        df[c] = df[c].apply(lambda x: x.replace("\n", ""))
    df['relations'] = df['relations'].apply(lambda x: x.replace("\n", "."))
    train_df = df[df['is_train'] == 1]
    print(f"36 Samples is too much. Removing one sample with label=3 from training set")
    ''' Remove only one sample that have LABEL_COL=3 '''
    train_df_not_3 = train_df[train_df[LABEL_COL] != 3]
    train_df_3 = train_df[train_df[LABEL_COL] == 3]
    ''' Take all except one '''
    train_df_3 = train_df_3.head(len(train_df_3) - 3)
    train_df = pd.concat([train_df_not_3, train_df_3])
    ''' Shuffle '''
    train_df = train_df.sample(frac=1)
    print(f"Train value counts ({len(train_df)} items)")
    print(train_df[LABEL_COL].value_counts(normalize=True))
    return train_df


def build_prompt_from_columns(r, columns):
    items = []
    for c in columns:
        if 'isAnalogy(0-3)' in c:
            item_str = f"LABEL: {int(r[c])}"
        elif 'refined comment' in c.lower():
            rationale = r[c].strip().rstrip('\n')
            item_str = f"{RATIONALE}: {rationale}"
        elif c == 'not analogy reasons':
            if int(r['isAnalogy(0-3)']) >= 2:
                item_str = f"{c.upper()}: nan"
            else:
                item_str = f"{c.upper()}: {r[c]}"
        elif c == 'analogy type':
            if r['isAnalogy(0-3)'] <= 1:
                item_str = f"{c.upper()}: nan"
            else:
                item_str = f"{c.upper()}: {r[c]}"
        else:
            item_str = f"{c.upper()}: {r[c]}"
        items.append(item_str)
    return "\n".join(items)


def get_gpt4_response(r_dict, r_prompt, max_tokens=100):
    try:
        response_gpt4 = predict_sample_openai_chatgpt(r_dict, r_prompt, max_tokens, engine="gpt-4")
    except Exception as ex:
        response_gpt4 = {}
        response_gpt4['prediction'] = None
        print(f"Exception {str(ex)}")
        return None
    return response_gpt4


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == '__main__':
    main()
