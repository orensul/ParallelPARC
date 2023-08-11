from gpt_utils import get_gpt_prediction
from utils import get_xlsx, get_txt, dump_csv


class DifferentSequenceOfEventsGenerator:

    def __init__(self, target_csv_path, results_path, start_range=0, end_range=-1):
        self.csv = get_xlsx(target_csv_path)
        self.start_range = start_range
        self.end_range = end_range if end_range > -1 else len(self.csv['target_paragraph'])
        self.first_prompt = get_txt(r'prompts/one_shot_events_new_order.txt')
        self.second_prompt = get_txt(r'prompts/few_shot_generate_new_paragraph.txt')
        self.results_path = results_path
        self.model = 'gpt-4'

    # The function receives the paragraph type for which it will create distractors
    def generate_distractors(self, paragraph_type='target'):

        # paragraphs in the range provided
        target_paragraphs_in_range = list(self.csv[f'{paragraph_type}_paragraph'].values())[self.start_range:self.end_range]

        for target_paragraph_index in range(0, len(self.csv[f'{paragraph_type}_paragraph'])):
            target_paragraph_index = str(target_paragraph_index)
            target_paragraph = self.csv[f'{paragraph_type}_paragraph'][target_paragraph_index]

            if target_paragraph in target_paragraphs_in_range:
                # First prompt - 1) list the events order of the paragraph, 2) switches between two events to create an illogical events order 3) provide an explanation why the new events order is illogical
                first_prompt_results = get_gpt_prediction(self.first_prompt + target_paragraph, self.model, 'events_order_first')
                self.csv[f'{paragraph_type}_events_order'].update({target_paragraph_index: first_prompt_results['input_paragraph_events']})
                self.csv[f'{paragraph_type}_new_events_order'].update({target_paragraph_index: first_prompt_results['new_paragraph_events']})
                self.csv['explanation'].update({target_paragraph_index: first_prompt_results['event_replacement_explanation']})

                # Second prompt - 2) Create a new paragraph with the illogical sequence of events
                second_prompt = self.first_prompt + target_paragraph + '\n\n' + first_prompt_results['plain_response'] + '\n\n' + \
                                self.second_prompt + self.csv[f'{paragraph_type}_new_events_order'][target_paragraph_index]
                new_paragraph = get_gpt_prediction(second_prompt, self.model, 'events_order_second')
                self.csv['new_paragraph'].update({target_paragraph_index: new_paragraph})
            else:
                self.csv[f'{paragraph_type}_new_events_order'].update({target_paragraph_index: None})
                self.csv['explanation'].update({target_paragraph_index: None})
                self.csv['new_paragraph'].update({target_paragraph_index: None})

        dump_csv(self.results_path, self.csv)


# Generate Distractors
# DifferentSequenceOfEventsGenerator receives as input
# a) the source CSV file
# b) the result csv file name
# c) the start index of the source csv rows
# d) the end index of the source csv rows
different_sequence_of_events_generator = DifferentSequenceOfEventsGenerator(r'example_20_distractor_for_random_in_validation_set.csv', r'example_20_distractor_for_random_in_validation_set - results.csv', 0, 20)

# Generate distractors to the paragraph type that was provided
different_sequence_of_events_generator.generate_distractors('random')