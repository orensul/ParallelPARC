
import pandas as pd
import json


def get_accuracy_per_worker(df):
    accuracy_per_worker = df.groupby(['WorkerId'])['is_correct'].mean().round(2)
    accuracy_per_worker = accuracy_per_worker.reset_index()
    return accuracy_per_worker


def get_majority_vote(new_df):
    new_df['majority_prediction'] = new_df.groupby(['sample_id'])['answer'].transform(lambda x: x.mode()[0])
    new_df['is_correct_majority'] = new_df['majority_prediction'] == new_df['ground_truth']
    new_df['is_correct_majority'] = new_df['is_correct_majority'].apply(lambda x: 1 if x == True else 0)
    print("majority vote accuracy: " + str(new_df['is_correct_majority'].sum() / new_df['is_correct_majority'].count() * 100))
    return new_df


def convert_to_list(x):
    if isinstance(x, str):
        return json.loads(x)
    return x


def find_index(row, candidates_column_name='candidates_paragraphs_y', distractor_column_name='distractor_target_paragraph'):
    return row[candidates_column_name].index(row[distractor_column_name])


def mc_basic():
    gt_mc_basic = pd.read_csv('eval_mc_basic.csv')
    mturk_mc_basic = pd.read_csv('mturk_eval_mc_basic_merged_results.csv')
    mturk_mc_basic = mturk_mc_basic[['WorkerId', 'Input.candidates_paragraphs', 'Input.first_paragraph', 'Answer.target_paragraph_number', 'Answer.sample_id']]
    mturk_mc_basic = mturk_mc_basic.rename(columns={'Input.candidates_paragraphs': 'candidates_paragraphs', 'Input.first_paragraph' : 'first_paragraph', 'Answer.target_paragraph_number' : 'answer', 'Answer.sample_id' : 'sample_id'})
    mturk_mc_basic['answer'] = mturk_mc_basic['answer'].replace({1:'C1', 2:'C2', 3:'C3', 4:'C4'})
    merged_df = gt_mc_basic.merge(mturk_mc_basic, how='inner', on=['sample_id'])
    merged_df['is_correct'] = merged_df['answer'] == merged_df['ground_truth']
    merged_df['is_correct'] = merged_df['is_correct'].apply(lambda x: 1 if x == True else 0)
    grouped = merged_df.groupby('sample_id')['answer'].nunique()
    agreement_count = sum(grouped == 1)
    print('Number of sample_ids with full agreement:', agreement_count)
    print('Agreement ratio: ' + str( agreement_count / len(grouped)))
    majority_vote_df = merged_df.copy()
    majority_vote_df = get_majority_vote(majority_vote_df)
    acc_per_type_maj_vote = get_accuracy_per_worker(majority_vote_df)
    accuracy_per_worker = merged_df.groupby('WorkerId')['is_correct'].mean()
    std_dev = accuracy_per_worker.std()
    print(f"Standard Deviation: {std_dev}")
    return accuracy_per_worker, acc_per_type_maj_vote, majority_vote_df



def mc_advanced():
    gt_mc_advanced = pd.read_csv('eval_mc_advanced.csv')
    mturk_mc_advanced = pd.read_csv('mturk_eval_mc_advanced_merged_results.csv')


    mturk_mc_advanced = mturk_mc_advanced[['WorkerId', 'Input.candidates_paragraphs', 'Input.first_paragraph', 'Answer.target_paragraph_number', 'Answer.sample_id']]
    mturk_mc_advanced = mturk_mc_advanced.rename(columns={'Input.candidates_paragraphs': 'candidates_paragraphs', 'Input.first_paragraph' : 'first_paragraph', 'Answer.target_paragraph_number' : 'answer', 'Answer.sample_id' : 'sample_id'})
    mturk_mc_advanced['answer'] = mturk_mc_advanced['answer'].replace({1:'C1', 2:'C2', 3:'C3', 4:'C4'})

    merged_df = gt_mc_advanced.merge(mturk_mc_advanced, how='inner', on=['sample_id'])
    merged_df['is_correct'] = merged_df['answer'] == merged_df['ground_truth']
    merged_df['is_correct'] = merged_df['is_correct'].apply(lambda x: 1 if x == True else 0)

    grouped = merged_df.groupby('sample_id')['answer'].nunique()
    print("Number of unique sample_ids: " + str(len(grouped)))
    agreement_count = sum(grouped == 1)
    print('Number of sample_ids with full agreement:', agreement_count)
    print('Agreement ratio: ' + str( agreement_count / len(grouped)))
    majority_vote_df = merged_df.copy()
    majority_vote_df = get_majority_vote(majority_vote_df)
    acc_per_type_maj_vote = get_accuracy_per_worker(majority_vote_df)
    accuracy_per_worker = merged_df.groupby('WorkerId')['is_correct'].mean()
    std_dev = accuracy_per_worker.std()
    print(f"Standard Deviation: {std_dev}")
    return accuracy_per_worker, acc_per_type_maj_vote, majority_vote_df


def get_percentage_wrong_due_distractor(majority_vote_df):
    df_correct_0 = majority_vote_df[majority_vote_df['is_correct_majority'] == 0]
    df_correct_0['candidates_paragraphs_y'] = df_correct_0['candidates_paragraphs_y'].apply(convert_to_list)
    df_correct_0['distractor_index'] = df_correct_0.apply(find_index, axis=1)
    df_correct_0['distractor_index'] += 1
    df_correct_0['distractor_index'] = 'C' + df_correct_0['distractor_index'].astype(str)
    num_prediction_distractor = df_correct_0[df_correct_0['majority_prediction'] == df_correct_0['distractor_index']].shape[0]
    total_correct_0 = df_correct_0.shape[0]
    percentage = (num_prediction_distractor / total_correct_0) * 100
    print("Num incorrect prediction on distractor: " + str(num_prediction_distractor))
    print("Num incorrect prediction on distractor divided by the number of workers: " + str(num_prediction_distractor / 3))
    print("Num of incorrect: " + str(total_correct_0))
    print("Num of incorrect divided by the number of workers: " + str(total_correct_0 / 3))
    print(f"The percentage of 'is_correct' 0 where 'prediction' equals 'distractor_target_paragraph' is {percentage}%.")






def main():
    # print("--- eval humans mc basic ---")
    # accuracy_per_worker, acc_per_type_maj_vote, majority_vote_df = mc_basic()
    print("--- eval humans mc advanced ---")
    accuracy_per_worker, acc_per_type_maj_vote, majority_vote_df = mc_advanced()
    majority_vote_df.to_csv('test.csv')
    # get_percentage_wrong_due_distractor(majority_vote_df)



if __name__ == '__main__':
    main()

