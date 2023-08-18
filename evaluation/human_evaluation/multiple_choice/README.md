This folder consists of the files related to the human evaluation of the multiple-choice task (supervised settings).
See Section 5 -- "Evaluating Humans and LLMs" in the paper <br>

* Basic setup: "eval_mc_basic.csv" includes the data for evaluation, "mturk_eval_mc_basic_merged_results.csv" includes the annotations of the workers. <br>
In total, 25 samples, 100% accuracy (majority vote), 88% agreement (std=0.04). <br>
* Advanced setup: "eval_mc_advanced.csv" includes the data for evaluation, "mturk_eval_mc_advanced_merged_results.csv" includes the annotations of the workers. <br>
In total, 50 samples, 96% accuracy (majority vote), 66% agreement (std=0.04). <br> Errors due to choosing the distractors (100% of the cases) <br>
* The code "eval_humans_mc.py" run this evaluation and prints the results.
* We REDACTED from the code any information about the Workers of Amazon Mechanical Turk for anonymity. <br>
* We REDACTED from the code any information about the authors of the paper for anonymity. <br>
