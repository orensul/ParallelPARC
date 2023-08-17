import pandas as pd

from sklearn.metrics import accuracy_score


def clean_string(s):
    s = ' '.join(s.split())  # Remove extra spaces
    s = s.replace('\n', ' ')  # Replace new line characters with space
    return s.lower() # #lowercase




def get_mturk_result_df(list_of_file_paths):
    dataframes = []
    for path in list_of_file_paths:
        print(path)
        dataframes.append(pd.read_csv(path))
    mturk_df = pd.concat(dataframes)
    mturk_df.reset_index(drop=True, inplace=True)
    mturk_df = mturk_df[mturk_df['WorkerId'].isin(['A9HQ3E0F2AGVO', 'A2R1GDWV4RLIUY', 'A3RVHUY67SVXQV'])]
    mturk_df = mturk_df[['WorkerId', 'Input.source_paragraph', 'Input.target_paragraph', 'Answer.analogy_flag.analogy', 'Answer.analogy_flag.not_analogy']]
    mturk_df['prediction'] = mturk_df['Answer.analogy_flag.analogy'].apply(lambda x: 1 if x == True else 0)
    mturk_df = mturk_df[['WorkerId', 'Input.source_paragraph', 'Input.target_paragraph', 'prediction']]
    mturk_df = mturk_df.rename(columns={'Input.source_paragraph': 'source_paragraph', 'Input.target_paragraph' : 'target_paragraph'})
    mturk_df['source_paragraph'] = mturk_df['source_paragraph'].apply(clean_string)
    mturk_df['target_paragraph'] = mturk_df['target_paragraph'].apply(clean_string)
    return mturk_df

def get_gt_df(list_of_file_paths):
    dataframes = []
    for path in list_of_file_paths:
        dataframes.append(pd.read_csv(path))
    gt_df = pd.concat(dataframes)
    gt_df.reset_index(drop=True, inplace=True)
    gt_df['source_paragraph'] = gt_df['source_paragraph'].apply(clean_string)
    gt_df['target_paragraph'] = gt_df['target_paragraph'].apply(clean_string)
    gt_df = gt_df.drop_duplicates(subset=['source_paragraph' , 'target_paragraph'])
    gt_df['type'] = gt_df['type'].replace(['far analogy', 'close analogy'], 'analogy')
    return gt_df




def calculate_accuracy(df):
    return accuracy_score(df['ground_truth'], df['majority_prediction'])



def get_accuracy_per_type(df):
    accuracy_per_worker_and_type = df.groupby(['WorkerId', 'type'])['is_correct'].mean().round(2)
    accuracy_per_worker_and_type_df = accuracy_per_worker_and_type.reset_index()
    return accuracy_per_worker_and_type_df


def get_majority_vote(new_df):
    new_df['majority_prediction'] = new_df.groupby(['source_paragraph', 'target_paragraph'])['prediction'].transform(lambda x: x.mode()[0])
    new_df['is_correct_majority'] = new_df['majority_prediction'] == new_df['ground_truth']
    new_df['is_correct_majority'] = new_df['is_correct_majority'].apply(lambda x: 1 if x == True else 0)
    print("majority vote accuracy: " + str(new_df['is_correct_majority'].sum() / new_df['is_correct_majority'].count() * 100))
    return new_df


def zero_shot_exp():
    mturk_df = pd.read_csv('mturk_results_zero_shot.csv')
    gt_df = get_gt_df(['mturk_for_eval_zero_shot.csv'])
    merged_df = gt_df.merge(mturk_df, how='inner', on=['source_paragraph', 'target_paragraph'])
    merged_df['is_correct'] = merged_df['prediction'] == merged_df['ground_truth']
    merged_df['is_correct'] = merged_df['is_correct'].apply(lambda x: 1 if x == True else 0)

    grouped = merged_df.groupby(['source_paragraph', 'target_paragraph'])['prediction'].std()
    agreement_count = sum(grouped == 0)
    print('Number of sample_ids with full agreement:', agreement_count)
    unique_sample_ids = merged_df['sample_id'].nunique()
    print('Number of unique sample_ids:', unique_sample_ids)
    print("agreement ratio: " + str(agreement_count / unique_sample_ids))

    majority_vote_df = merged_df.copy()
    majority_vote_df = get_majority_vote(majority_vote_df)
    accuracy_per_type = majority_vote_df.groupby('type').apply(calculate_accuracy)
    acc_per_type_maj_vote = get_accuracy_per_type(majority_vote_df)
    accuracy_per_worker = merged_df.groupby('WorkerId')['is_correct'].mean()
    accuracy_per_worker_and_type = get_accuracy_per_type(merged_df)
    print(accuracy_per_worker)
    std_dev = accuracy_per_worker.std()
    print(f"Standard Deviation: {std_dev}")
    return accuracy_per_type, acc_per_type_maj_vote, accuracy_per_worker, accuracy_per_worker_and_type





def supervised_exp():
    mturk_df = pd.read_csv('mturk_results_supervised.csv')
    gt_df = get_gt_df(['mturk_for_eval_supervised.csv'])

    merged_df = gt_df.merge(mturk_df, how='inner', on=['source_paragraph', 'target_paragraph'])
    merged_df['is_correct'] = merged_df['prediction'] == merged_df['ground_truth']
    merged_df['is_correct'] = merged_df['is_correct'].apply(lambda x: 1 if x == True else 0)


    grouped = merged_df.groupby(['source_paragraph', 'target_paragraph'])['prediction'].std()
    agreement_count = sum(grouped == 0)
    print('Number of sample_ids with full agreement:', agreement_count)
    unique_sample_ids = merged_df['sample_id'].nunique()
    print('Number of unique sample_ids:', unique_sample_ids)
    print("agreement ratio: " + str(agreement_count / unique_sample_ids))

    majority_vote_df = merged_df.copy()
    majority_vote_df = get_majority_vote(majority_vote_df)
    accuracy_per_type = majority_vote_df.groupby('type').apply(calculate_accuracy)
    acc_per_type_maj_vote = get_accuracy_per_type(majority_vote_df)


    accuracy_per_worker = merged_df.groupby('WorkerId')['is_correct'].mean()
    accuracy_per_worker_and_type = get_accuracy_per_type(merged_df)
    std_dev = accuracy_per_worker.std()
    print(f"Standard Deviation: {std_dev}")
    return accuracy_per_type, acc_per_type_maj_vote, accuracy_per_worker, accuracy_per_worker_and_type



def main():
    print('--- Zero-shot Humans ---')
    accuracy_per_type, acc_per_type_maj_vote, accuracy_per_worker, accuracy_per_worker_and_type = zero_shot_exp()
    print('--- Supervised setting Humans ---')
    accuracy_per_type, acc_per_type_maj_vote, accuracy_per_worker, accuracy_per_worker_and_type = supervised_exp()


if __name__ == '__main__':
    main()
