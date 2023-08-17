This folder contains all the files related to fine-tuning FlanT5-small for the binary task, by training it on the siver-set. See RQ2 in Section 5 -- "Evaluating Humans and LLMs" in the paper. <br>
* The folder "test_sets" contains a jsonl file with the 620 instances from the gold-test-set (310 analogies, 155 simple negatives (random), 155 challenging negatives (distractor)). <br>
* The folder "train_samples" contains a jsonl file with the 620 instances from the silver-training-set (again with the same distribution). <br>
* The file "binary_prompt.txt" contains the prompt given in zero-shot for the models in the binary task. <br>
* The code in "flan_t5_ft.py". <br>
* The file "fine_tuning_results.csv" contains the results after running the code (the inference step).
