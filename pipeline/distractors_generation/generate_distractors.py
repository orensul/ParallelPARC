
import pandas as pd



def create_analogies_w_challenging_distractor_csv(set, distractor_csv_path, output_csv_path):
    distractors_df = pd.read_csv(distractor_csv_path)
    distractors_df['new_paragraph'] = distractors_df['new_paragraph'].str.replace(r'\[\d\]', '', regex=True)
    distractors_df['new_paragraph'] = distractors_df['new_paragraph'].str.replace("  ", " ")

    if set == 'gold':
        distractors_df['analogy_type'] = distractors_df['analogy_type'].str.lower()
        distractors_df = distractors_df[['sample_id', 'source_id', 'source_subject', 'source_domain', 'target_domain',
                                         'target_subject', 'target_field', 'relations', 'source_paragraph', 'target_paragraph',
                                         'analogy_type', 'new_paragraph']]
    else:
        distractors_df = distractors_df[['sample_id', 'source_id', 'source_subject', 'source_domain', 'target_domain',
                                         'target_subject', 'target_field', 'relations', 'source_paragraph', 'target_paragraph', 'new_paragraph']]


    distractors_df = distractors_df.rename(columns={'new_paragraph': 'distractor_target_paragraph'})
    print(len(distractors_df))
    distractors_df.to_csv(output_csv_path)





def main():
    create_analogies_w_challenging_distractor_csv('gold', 'gold_set_distractors/distractors.csv', '../../datasets/gold_test_set/gold_set_analogies_w_challenging_distractors.csv')
    create_analogies_w_challenging_distractor_csv('silver', 'silver_set_distractors/distractors.csv', '../../datasets/silver_training_set/silver_set_analogies_w_challenging_distractors.csv')



if __name__ == '__main__':
    main()