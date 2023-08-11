
import pandas as pd


def get_accuracy_per_type(df):
    percentage_df = df.groupby(['model', 'type'])['is_correct'].sum() / df.groupby(['model', 'type'])['is_correct'].count() * 100
    percentage_df = percentage_df.reset_index()
    return percentage_df


# Define a function to calculate the percentage
def calculate_percentage(df):
    num_prediction_distractor = df[df['prediction'] == df['distractor_target_paragraph']].shape[0]
    total_correct_0 = df.shape[0]
    return (num_prediction_distractor / total_correct_0) * 100


def main():
    print('--- Binary task zero-shot models results per sample type (analogy, distractor, random)')
    results = pd.read_csv('proparalogy/experiments_results/data_for_eval_balanced_shuffled_binary_task_predictions_metadata.csv')
    results['is_correct'] = results['prediction'] == results['ground_truth']
    results['is_correct'] = results['is_correct'].apply(lambda x: 1 if x == True else 0)
    results['type'] = results['type'].replace(['far analogy', 'close analogy'], 'analogy')
    accuracy_per_model_and_type_df = get_accuracy_per_type(results)
    print(accuracy_per_model_and_type_df)

    print('--- Binary task supervised models results per sample type (analogy, distractor, random)')
    results = pd.read_csv('proparalogy/experiments_results/data_for_eval_balanced_shuffled_binary_task_few_shot_predictions_metadata.csv')
    results['is_correct'] = results['prediction'] == results['ground_truth']
    results['is_correct'] = results['is_correct'].apply(lambda x: 1 if x == True else 0)
    results['type'] = results['type'].replace(['far analogy', 'close analogy'], 'analogy')
    accuracy_per_model_and_type_df = get_accuracy_per_type(results)
    print(accuracy_per_model_and_type_df)

    print('--- Multiple-choice zero-shot models distractor mistakes percentage')
    results = pd.read_csv('proparalogy/experiments_results/data_for_eval_random_distractor_multiple_choice_task_predictions_metadata.csv')
    results['is_correct'] = results['prediction_index'] == results['ground_truth']
    results['is_correct'] = results['is_correct'].apply(lambda x: 1 if x == True else 0)
    results['analogy_type'] = results['analogy_type'].replace(['far analogy', 'close analogy'], 'analogy')
    results_grouped = results[results['is_correct'] == 0].groupby('model').apply(calculate_percentage)
    print(results_grouped)





if __name__ == '__main__':
    main()