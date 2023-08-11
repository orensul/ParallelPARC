

import pandas as pd

# See Section 3.4 Human validation (Annotation process results) in the paper
def main():
    df_gt = pd.read_csv('mturk_data.csv')
    df_gt = df_gt[df_gt['sum_vote_analogy'].notnull()]
    agreement_ratio = df_gt[df_gt["sum_vote_analogy"].isin([0, 3])].shape[0] / len(df_gt)
    ratio_analogies_strict = df_gt['sum_vote_analogy'].isin([3]).sum() / len(df_gt)
    ratio_analogies_majority_vote = df_gt['sum_vote_analogy'].isin([2, 3]).sum() / len(df_gt)
    ratio_further_inspection = df_gt['sum_vote_analogy'].isin([0]).sum() / len(df_gt)
    ratio_one_annotators = df_gt['sum_vote_analogy'].isin([1]).sum() / len(df_gt)
    ratio_two_annotators = df_gt['sum_vote_analogy'].isin([2]).sum() / len(df_gt)

    print("total annotated data size: " + str(len(df_gt)))
    print("ratio all agree: " + str(agreement_ratio))
    print("ratio analogies strict: " + str(ratio_analogies_strict))
    print("ratio analogies majority vote: " + str(ratio_analogies_majority_vote))
    print("ratio further inspection: " + str(ratio_further_inspection))
    print("ratio one annotator vote for analogy: " + str(ratio_one_annotators))
    print("ratio one annotators vote for analogy: " + str(ratio_two_annotators))


if __name__ == '__main__':
    main()