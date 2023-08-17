This folder is related to Section 3.4 --  Human Validation in the paper. <br>

* The folder "auto_labeling_predictions" includes all 8 batches given to the auto labeling model in order to construct
the gold-set. <br> In total 1859 instances (out of them 828 validated by humans -- predicted with score 0 or 3). <br>
* The file "mturk_data.csv" includes all analogy candidates after annotation (958 annotations in total). <br>
* The code in "annotation_process_results.py" includes some statistics about the annotated data, and the annotators agreement (78.6%). <br>
* The code in "auto_labeling_evaluation.py" includes the results from evaluation (accuracy of 85.14% and f1-score of 83.45% with 79.5% precision on the positive class (analogy) and 90% on the negative). <br>
<br>
This folder also related to Section 4 -- Dataset Analysis in the paper. <br>
* The code in "dataset_analysis.py" includes all statistics about our gold and silver sets. <br>
* In addition, it includes the distribution of different type of analogies in the gold-set (Table 1)
as well as statistics about the different reasons for "further inspection" (Table 2)

