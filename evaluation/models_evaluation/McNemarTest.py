
import pandas as pd
from statsmodels.sandbox.stats.runs import mcnemar

def write_csv_file_for_mcnemar_test():
    binary_task_results = pd.read_csv('proparalogy/experiments_results/data_for_eval_balanced_shuffled_binary_task_predictions_metadata.csv')
    filtered_binary_task_results = binary_task_results[binary_task_results['model'] == "FLAN-T5-xxl"]
    filtered_binary_task_results = filtered_binary_task_results[['sample_id', 'type', 'prediction_index', 'ground_truth']]
    filtered_binary_task_results['type'] = filtered_binary_task_results['type'].replace(['close analogy', 'far analogy'], 'analogy')
    fine_tuning_results = pd.read_csv('proparalogy_fine_tune_flanT5Small/fine_tuning_results.csv')
    fine_tuning_results = fine_tuning_results[['sample_id', 'type', 'predictions', 'ground_truth']]
    merged_data = pd.merge(filtered_binary_task_results, fine_tuning_results, how='inner', on=['sample_id', 'type'])
    merged_data.rename(columns={
      'prediction_index': 'flan_t5_xxl_zero_shot_prediction',
      'predictions': 'flan_t5_small_ft_prediction',
      'ground_truth_x': 'ground_truth'}, inplace=True)
    merged_data.drop(columns=['ground_truth_y'], inplace=True)
    merged_data = merged_data[['sample_id', 'type', 'flan_t5_xxl_zero_shot_prediction', 'flan_t5_small_ft_prediction', 'ground_truth']]
    merged_data.to_csv('proparalogy/experiments_results/flan_t5_xxl_zero_shot_vs_flan_t5_small_ft.csv', index=False)



def get_model_prediction_and_gt(df, model_name):
    df = df[df['model'] == model_name]
    df = df[['sample_id', 'prediction_index', 'ground_truth']]
    return df

def get_merge_results_for_model(df1, df2):
    merged_data = pd.merge(df1, df2, how='inner', on=['sample_id'])
    merged_data.rename(columns={
        'prediction_index_x': 'prediction_basic',
        'prediction_index_y': 'prediction_advanced',
        'ground_truth_x': 'ground_truth_basic',
        'ground_truth_y': 'ground_truth_advanced'}, inplace=True)
    return merged_data

def mcnemar_test(table):
    test_result = mcnemar(table)
    _, p_value = test_result
    print(f'p-value: {p_value}')

def main():
    print('--- McNemar test for RQ2 ---')
    write_csv_file_for_mcnemar_test()
    df = pd.read_csv('proparalogy/experiments_results/flan_t5_xxl_zero_shot_vs_flan_t5_small_ft.csv')
    table = pd.crosstab(df['flan_t5_xxl_zero_shot_prediction'] == df['ground_truth'],
                        df['flan_t5_small_ft_prediction'] == df['ground_truth'])
    mcnemar_test(table)

    print('--- McNemar test for RQ3 ---')
    basic_setup_results = pd.read_csv('proparalogy/experiments_results/data_for_eval_random_candidates_multiple_choice_task_predictions_metadata.csv')
    advanced_setup_results = pd.read_csv('proparalogy/experiments_results/data_for_eval_random_distractor_multiple_choice_task_predictions_metadata.csv')
    for model_name in ['GPT4', 'ChatGPT', 'FLAN-T5-xxl', 'FLAN-T5-xl']:
        basic_results = get_model_prediction_and_gt(basic_setup_results, model_name)
        advanced_results = get_model_prediction_and_gt(advanced_setup_results, model_name)
        df = get_merge_results_for_model(basic_results, advanced_results)
        print("p-value for " + model_name)
        table = pd.crosstab(df['prediction_basic'] == df['ground_truth_basic'],
                            df['prediction_advanced'] == df['ground_truth_advanced'])
        mcnemar_test(table)





if __name__ == '__main__':
    main()