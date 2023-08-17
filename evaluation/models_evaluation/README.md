This folder contains all files related to the models evaluation in both the binary classification and multiple-choice task. <br>
See Section 5 -- "Evaluating Humans and LLMs" in the paper. <br>

* "proparalogy" folder contains all the relevant files for the zero-shot model's evaluation (for both tasks). <br> 
Also, it contains files related to the few-shot for GPT4 (supervised settings) and comparison between FlanT5-xxl in zero-shot to FlanT5-small supervised. <br>
* "proparalogy_fine_tune_flanT5Small" folder contains all files related to the fine-tuning of FlanT5-small for the binary task. <br>
* "eval_results_analysis.py" prints all the evaluation results (summarized in Table 3 in the paper (Section 5)). <br>
* "McNemarTest.py" prints the p-values under McNemar test (0.05 level with Bonferroni correction) for RQ1 and RQ2. <br>
* "ProParaLogy_models_zero_shot_exp.ipynb" is a google colab notebook that runs all the zero-shot experiments. <br>
You should create the folder "proparalogy" in your Google Drive, with the same sub-folders and files as here. <br>
* "create_evaluation_data.py" has the code for generating data for evaluation, but you can ignore it (all relevant files are already in the folders). <br>