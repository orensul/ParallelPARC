import pandas as pd


csv_result_no_paragraphs_prompt = 'multiple_choice_task_prompt_no_paragraphs_predictions_metadata.csv'
csv_result_w_paragraphs_prompt = 'multiple_choice_task_prompt_w_paragraphs_predictions_metadata.csv'



def get_models_performance_df(file):
    df = pd.read_csv(file)
    correct_counts_by_model = df.groupby('model').apply(lambda x: (x['prediction'] == x['ground_truth']).sum())
    total_counts_by_model = df['model'].value_counts()

    accuracy_df = pd.DataFrame({
        'correct_counts': correct_counts_by_model,
        'total_counts': total_counts_by_model,
        'accuracy' : correct_counts_by_model / total_counts_by_model
    })
    return accuracy_df


def error_analysis(file):
    df = pd.read_csv(file)
    mismatch_df = df[df['prediction'] != df['ground_truth']]
    distractor_failures = mismatch_df.groupby('model').apply(lambda x: (x['prediction'] == x['distractor_target_paragraph']).sum())
    total_failures = mismatch_df['model'].value_counts()


    error_analysis_df =  pd.DataFrame({
        'distractor_failures' : distractor_failures,
        'total_failures' : total_failures,
        'percentage' : distractor_failures / total_failures
    })
    return error_analysis_df


if __name__ == '__main__':
    for file in [csv_result_w_paragraphs_prompt, csv_result_no_paragraphs_prompt]:
        print("file name: " + file)
        acc_df = get_models_performance_df(file)
        print("models performance results:")
        print(acc_df)

        error_analysis_df = error_analysis(file)
        print("failures due to distractor analysis:")
        print(error_analysis_df)