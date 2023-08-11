
import pandas as pd


def get_analogy_type_ratio(all_data, min_vote):
    if min_vote == 3:
        print("vote = 3")
        analogies = all_data[all_data['sum_vote_analogy'].isin([3])]
    elif min_vote >= 2:
        print("vote >= 2")
        analogies = all_data[all_data['sum_vote_analogy'].isin([2, 3])]
    elif min_vote >= 1:
        print("vote >= 1")
        analogies = all_data[all_data['sum_vote_analogy'].isin([1, 2, 3])]

    print("#Samples = " + str(len(analogies)))

    analogies['analogy_type'] = analogies['analogy_type'].str.lower()
    result = analogies['analogy_type'].value_counts()

    total = result.sum()
    close_analogy_ratio = result['close analogy'] / total
    far_analogy_ratio = result['far analogy'] / total
    print("Close analogy ratio:", close_analogy_ratio * 100)
    print("Far analogy ratio:", far_analogy_ratio * 100)


def get_reasons_ratio(all_data, max_vote):
    if max_vote == 0:
        print("vote = 0")
        not_analogies = all_data[all_data['sum_vote_analogy'].isin([0])]
    elif max_vote <= 1:
        print("vote <= 1")
        not_analogies = all_data[all_data['sum_vote_analogy'].isin([0, 1])]
    elif max_vote <= 2:
        print("vote <= 2")
        not_analogies = all_data[all_data['sum_vote_analogy'].isin([0, 1, 2])]

    print("#Samples = " + str(len(not_analogies)))

    filtered_df = not_analogies[not_analogies['not_analogy_reasons'].notna()]
    filtered_df['not_analogy_reasons'] = filtered_df['not_analogy_reasons'].str.lower()

    other_df = filtered_df[filtered_df['not_analogy_reasons'].str.contains('not analogy - other')]
    dissimilar_relations_df = filtered_df[filtered_df['not_analogy_reasons'].str.contains('not analogy - dissimilar relation')]
    cyclic_non_cyclic_df = filtered_df[filtered_df['not_analogy_reasons'].str.contains('not analogy - cyclic/non-cyclic')]
    misinfo_df = filtered_df[filtered_df['not_analogy_reasons'].str.contains('not analogy - misinformation')]

    print("ratio dissimilar relations reason: " + str(round(len(dissimilar_relations_df) / len(not_analogies) * 100, 2)))
    print("ratio misinformation reason: " + str(round(len(misinfo_df) / len(not_analogies) * 100, 2)))
    print("ratio cyclic/non-cyclic reason: " + str(round(len(cyclic_non_cyclic_df) / len(not_analogies) * 100, 2)))
    print("ratio other reason: " + str(round(len(other_df) / len(not_analogies) * 100, 2)))


# See Section 4 Dataset Analysis in the paper
def main():
    all_data = pd.read_csv('mturk_data.csv')
    all_data = all_data[all_data['sum_vote_analogy'].notnull()]

    gold_set_analogies = pd.read_csv('../../datasets/gold_test_set/gold_set_analogies.csv')
    print("gold-set analogies size: " + str(len(gold_set_analogies)))

    silver_set_analogies = pd.read_csv('../../datasets/silver_training_set/silver_set_analogies.csv')
    print("silver-set analogies size: " + str(len(silver_set_analogies)))

    gold_set_further_inspection = pd.read_csv('../../datasets/gold_set_candidates_for_further_inspection.csv')
    print("gold-set further inspection size: " + str(len(gold_set_further_inspection)))

    silver_set_further_inspection = pd.read_csv('../../datasets/silver_set_candidates_for_further_inspection.csv')
    print("silver-set further inspection size: " + str(len(silver_set_further_inspection)))

    print("further inspection size: " + str(len(gold_set_further_inspection) + len(silver_set_further_inspection)))

    # This will print the results in Table 1 in the paper (The distribution of analogy types)
    get_analogy_type_ratio(all_data, 3)
    print("---")
    get_analogy_type_ratio(all_data, 2)
    print("---")
    get_analogy_type_ratio(all_data, 1)

    # This will print the results in Table 2 in the paper (The distribution of the different issues raised by annotators)
    get_reasons_ratio(all_data, 0)
    print('---')
    get_reasons_ratio(all_data, 1)
    print('---')
    get_reasons_ratio(all_data, 2)


if __name__ == '__main__':
    main()
