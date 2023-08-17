This folder contains all the relevant files for the zero-shot model's evaluation (for both tasks). <br> 
Also, it contains files related to the few-shot for GPT4 (supervised settings) and comparison between FlanT5-xxl in zero-shot to FlanT5-small supervised. <br>
See Section 5 -- "Evaluating Humans and LLMs" in the paper. <br>

* The folder "experiments" contains the csv files with data for evaluation (created by run "create_evaluation_data.py") <br>
* The folder "experiments_results" contains the metadata csv files with the predictions and GT of the evaluations (used by "eval_results_analysis.py" for analysis). <br>
The results of GPT4 in few-shot for the binary task appears in this file: "data_for_eval_balanced_shuffled_binary_task_few_shot_predictions_metadata.csv". <br>
The results of FlanT5-small after fine-tuning comparing to FlanT5-xxl in zero-shot, in file: "flan_t5_xxl_zero_shot_vs_flan_t5_small_ft.csv". <br>
* The folder "prompts" contains the prompts in zero-shot given to the models in both binary and multiple-choice tasks. <br> 
In addition, the file "GPT4_binary_prompt_mistakes_only.txt" is the few-shot prompts with the 5 mistakes of GPT4 from zero-shot.
* The notebook "ProParaLogy_models_zero_shot_exp.ipynb" run on this folder (Google colab pro+). <br>
Please add this folder to your Google Drive. <br>
* The code including a caching mechanism to save calls for GPT models (which cost money). Please put your OPEN_AI_KEY.