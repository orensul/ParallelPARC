import re

import openai

from utils import get_json, dump_json

API_KEY = ""
openai.api_key = API_KEY
GPT_CACHE = get_json(r'gpt_response_cache.json')
TEMPERATURE = -1  # Default 1

# Documentation
# https://platform.openai.com/docs/api-reference/making-requests

def extract_data_from_first_prompt_response(response):
    splited_response = response.split('\n')
    input_paragraph_events = '\n'.join(splited_response[splited_response.index('INPUT_PARAGRAPH_ORDER:')+1:splited_response.index('NEW_PARAGRAPH_ORDER:')]).strip()
    explanation_index = splited_response.index({'EXPLANATION:', 'EXPLANATION: ', 'EXPLANATION:  '}.intersection(splited_response).pop()) # sometimes EXPLENATION appears with multiple spaces
    new_paragraph_events = '\n'.join(splited_response[splited_response.index('NEW_PARAGRAPH_ORDER:')+1:explanation_index]).strip()
    event_replacement_explanation = '\n'.join(splited_response[explanation_index+1:]).strip()
    assert None not in [input_paragraph_events, new_paragraph_events, event_replacement_explanation]
    return {'input_paragraph_events': input_paragraph_events, 'new_paragraph_events': new_paragraph_events,
            'event_replacement_explanation': event_replacement_explanation, 'plain_response': response}

def extract_data_from_second_prompt_response(response):
    splitted_response = response.split('\n')
    splitted_response = list(filter(lambda paragraph: ('OUTPUT_PARAGRAPH'.lower() not in paragraph.lower() and 'OUTPUTS:'.lower() not in paragraph.lower()) and not paragraph.startswith('Event'), splitted_response))
    return ' '.join(splitted_response).strip()

def extract_response_data(response, prompt_type):
    if prompt_type == 'events_order_first':
        return extract_data_from_first_prompt_response(response)
    elif prompt_type == 'events_order_second':
        return extract_data_from_second_prompt_response(response)
    else:
        raise Exception('No parsing method found for prompt type')

def get_gpt_prediction(prompt, model, prompt_type):
    global TEMPERATURE
    if prompt_type == 'events_order_second':
        TEMPERATURE = 0.00001
    else:
        TEMPERATURE = -1

    cached_response = get_response_from_cache(prompt, model)
    if cached_response is not None:
        cached_response = cached_response.get('choices')[0].get('message').get('content')
        return extract_response_data(cached_response, prompt_type)

    try:
        output = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=abs(TEMPERATURE),
        )
    except:
        output = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=abs(TEMPERATURE),
        )
    save_gpt_response_in_cache(prompt, model, output)

    output_content = output.get('choices')[0].get('message').get('content')
    return extract_response_data(output_content, prompt_type)

def encode_key(key):
    return re.sub(r'[^\w\s]', '', key).replace('\r', '')

def get_gpt_cache_id(prompt, model):
    return encode_key(f"{prompt}___{model}___{TEMPERATURE}")

def get_response_from_cache(prompt, model):
    id = get_gpt_cache_id(prompt, model)
    return GPT_CACHE.get(id)

def save_gpt_response_in_cache(prompt, model, response):
    GPT_CACHE.update({get_gpt_cache_id(prompt, model): response})
    dump_json('gpt_response_cache.json', GPT_CACHE)

