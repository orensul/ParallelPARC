This folder is related to Section 3.1  Analogy Candidates Generation in the paper.

* The folder "analogy_candidates_dataset" includes the analogy candidates generated data (by running GPT-3.5).
The csv "GPT3_analogies_propara_topics_relations_mappings.csv" includes all 4680 candidates, and
the csv "GPT3_analogies_propara_topics_relations_filtered.csv" includes the 4288 candidates 
(after filtering candidates with less than 3 relations).

* The folder "prompts" includes the two separate prompts we supplied to GPT-3.5 for the generation.

* The Python code for the generation (employing these prompts): "generate_analogy_candidates.py"
