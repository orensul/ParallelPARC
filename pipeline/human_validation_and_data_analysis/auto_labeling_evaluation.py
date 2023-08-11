import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def filter_batch_full_agreements(df):
    df_filtered = df[df['predicted_label'].isin([0, 3])]
    df_filtered = df_filtered[['sample_id', 'source_id', 'source_subject', 'source_domain', 'target_domain', 'target_subject', 'target_field', 'relations',
         'source_paragraph', 'target_paragraph', 'predicted_label']]
    print("length of batch before filter full agreements: " + str(len(df)) + " and after filtering: " + str(len(df_filtered)))
    return df_filtered


def plot_conf_matrix(df):
    cm = confusion_matrix(df["is_analogy_gt"], df["is_analogy_pred"])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    cm = confusion_matrix(df["is_analogy_gt"], df["is_analogy_pred"])
    return cm



def eval(df):
    cm = plot_conf_matrix(df)
    true_negative, false_positive, false_negative, true_positive  = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    false_positives_percentage = (false_positive / (false_positive + true_negative)) * 100
    true_positives_percentage = (true_positive / (true_positive + false_positive)) * 100
    false_negatives_percentage = (false_negative / (false_negative + true_positive)) * 100
    true_negatives_percentage = (true_negative / (true_negative + false_positive)) * 100

    correct_predictions = (df["is_analogy_gt"] == df["is_analogy_pred"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    print("Accuracy:", accuracy)
    print(f"False Positive: {false_positives_percentage:.2f}%")
    print(f"True Positive: {true_positives_percentage:.2f}%")
    print(f"False Negative: {false_negatives_percentage:.2f}%")
    print(f"True Negative: {true_negatives_percentage:.2f}%")

    report = classification_report(df["is_analogy_gt"], df["is_analogy_pred"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    f1 = f1_score(df["is_analogy_gt"].to_list(), df["is_analogy_pred"].to_list())
    print("f1 task: " + str(f1))


# See Section 3.4 Human validation (Filtering model evaluation results) in the paper
def main():
    df_gt = pd.read_csv('mturk_data.csv')
    df_gt = df_gt[df_gt['sum_vote_analogy'].notnull()]


    df_pred_batch_1 = pd.read_csv('auto_labeling_predictions/predictions_batch_1.csv')
    df_pred_batch_2 = pd.read_csv('auto_labeling_predictions/predictions_batch_2.csv')
    df_pred_batch_3 = pd.read_csv('auto_labeling_predictions/predictions_batch_3.csv')
    df_pred_batch_4 = pd.read_csv('auto_labeling_predictions/predictions_batch_4.csv')
    df_pred_batch_5 = pd.read_csv('auto_labeling_predictions/predictions_batch_5.csv')
    df_pred_batch_6 = pd.read_csv('auto_labeling_predictions/predictions_batch_6.csv')
    df_pred_batch_7 = pd.read_csv('auto_labeling_predictions/predictions_batch_7.csv')
    df_pred_batch_8 = pd.read_csv('auto_labeling_predictions/predictions_batch_8.csv')

    df_pred_batches = [df_pred_batch_1, df_pred_batch_2, df_pred_batch_3, df_pred_batch_4, df_pred_batch_5, df_pred_batch_6, df_pred_batch_7, df_pred_batch_8]
    df_pred_batches_concat = pd.concat(df_pred_batches)
    unique_sample_id_count = df_pred_batches_concat['sample_id'].nunique()
    print("Number of unique sample_id:", unique_sample_id_count)


    df_filtered_pred_batches = []
    for df_pred_batch in df_pred_batches:
        df_filtered_pred_batches.append(filter_batch_full_agreements(df_pred_batch))

    df_filtered_pred = pd.concat(df_filtered_pred_batches)
    len_all_filtered_batches = len(df_filtered_pred)

    len_batches_before_filter = len(df_pred_batch_1) + len(df_pred_batch_2) + len(df_pred_batch_3) + len(df_pred_batch_4) + len(
        df_pred_batch_5) + len(df_pred_batch_6) + len(df_pred_batch_7) + len(df_pred_batch_8)
    print("Length all batches for predictions before filtering: " + str(len_batches_before_filter))
    # the number of predicted samples is 843 for human validation, but due to some technical issue with Amazon Mechanical Turk, human validation was on 828 samples (see 3.4 Human Validation in the paper)
    print("length of all batches after filtering full agreements: " + str(len_all_filtered_batches))



    df = df_filtered_pred.merge(df_gt, on='sample_id')
    print("value counts sum vote analogy")
    print(df['sum_vote_analogy'].value_counts())
    print("value counts predicted label")
    print(df['predicted_label'].value_counts())
    df["is_analogy_gt"] = df["sum_vote_analogy"].apply(lambda x: 1 if x >= 2 else 0)
    df["is_analogy_pred"] = df["predicted_label"].apply(lambda x: 1 if x == 3 else 0)

    # accuracy: 85.14%, f1-score: 83.5%
    eval(df)




if __name__ == '__main__':
    main()
    # print(1)
    # silver_all_data = pd.read_csv('../../datasets/silver_set_all_data.csv')
    # print(len(silver_all_data))
    #
    # data_for_pred = pd.read_csv('../auto_labeling/auto_label_data/shuffled_data_for_prediction_2905.csv')
    # print(len(data_for_pred))
    #
    # auto_label_train = pd.read_csv('../auto_labeling/auto_label_data/auto_label_train.csv')
    # print(len(auto_label_train))


