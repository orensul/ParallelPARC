This folder is related to Section 3.5 -- Distractor Generation in the paper. <br>

* The folder "evaluation" includes a csv with 100 annotated samples of generated distractors (80 out of 90 are good, 10 cannot be distractor in the first place, accuracy of 89%) <br>
* The folder "gold_set_distractors" includes a csv with all 310 generated distractors for the gold-set. <br>
* The folder "silver_set_distractors" includes a csv with all 403 generated distractors for the silver-set. <br>
* The folder "prompts" includes the two prompts for generating the distractor (using GPT4). <br>
* The code "generate_different_sequence_of_events_distractor.py" is responsible to generate the distractors using the prompts. <br>
* The code "generate_distractors.py" generates the files of distractors in the dataset folder (both for gold and silver). By removing "[x]" in each line of the generated paragraph. <br>
* The code "gpt_utils.py" is helper for the main code in "generate_different_sequence_of_events_distractor.py" and is responsible for the calls to GPT4's API, and also working with caching. <br>
* The code "utils.py" is helper for the main code in "generate_different_sequence_of_events_distractor.py" as well (for working with json files).

