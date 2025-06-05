# this script was used to classify the articles within one dataset each
# using the Llama 3.1 8B Instruct model and inclusion and exclusion criteria

# LIBRARIES
import datetime
import pandas as pd
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import csv
import os
import time
import sys
import argparse
import logging
from tqdm import tqdm
import argparse

# SETTINGS FOR THE LLAMA INFERENCE
BATCH_SIZE = 16
MAX_NEW_TOKENS = 150

# DATASET
try:
    print('parsing dataset argument')
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="name of the dataset for classification")
    args = parser.parse_args()

    dataset = args.dataset
    print('dataset is ', dataset)
except:
    sys.exit('please provide the name of the dataset for classification')

# IMPORT THE DATAFRAME
if not os.path.isdir('./datasets'):
    sys.exit('Error: Could not find "datasets" directory. Exiting.')
else:
    file_path = f'./datasets/{dataset}.csv'
    print('reading dataset from path ', file_path)
    if not os.path.isfile(file_path):
        sys.exit(f'Error: Could not find file "{dataset}.csv". Exiting.')

DATAFRAME = pd.read_csv(file_path)
print('dataset read')

# IMPORT INLCUSION AND EXCLUSION CRITERIA
if not os.path.isdir('./criteria'):
    sys.exit('Error: Could not find "criteria" directory. Exiting.')
else:
    print('reading inclusion criteria')
    criteria_path = f'./criteria/{dataset}.txt'
    print('inclusion criteria read')
    if not os.path.isfile(criteria_path):
        sys.exit(f'Error: Could not find file "{dataset}.txt". Exiting.')

with open(criteria_path, 'r') as file:
    criteria = file.read().split('\n')

try:
    INCLUSION = criteria[1]
except:
    INCLUSION = ''

try:
    EXCLUSION = criteria[3]
except:
    EXCLUSION = ''

print('Found inclusion and exclusion criteria', end='\n\n')
print(f'Inclusion criteria comprise:\n{INCLUSION}', end='\n\n')
print(f'Exclusion criteria comprise:\n{EXCLUSION}')


# LOAD MODEL, TOKENIZER AND CONFIG
# requires locally downloaded  model weights through huggingface-cli
model_path = './models/Meta-Llama-3.1-8B-Instruct/'
print(f'model path is {model_path}')


if not os.path.isdir(model_path):
    sys.exit(
        f'Error: Could not find directory "{model_path} with model weights". Exiting.')
else:
    print('model path found')

try:
    print('initializing tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left'
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    print('tokenizer initialized')
except:
    sys.exit('could not initialize tokenizer. Exiting.')


try:
    print('initializing model config')
    config = AutoConfig.from_pretrained(model_path)
    print('model config initialized')
except:
    sys.exit('could not initialize model config. Exiting.')


try:
    print('initializing model')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',  # automatic assignment of gpus,
        torch_dtype=torch.float16,
        pad_token_id=tokenizer.eos_token_id,
    )
    print('model initialized')
except:
    sys.exit('Could not initialize model. Exiting.')


# DISABLE RANDOMNESS IN INFERENCE
# Llama default configuration has some degree of randomness
model.generation_config.do_sample = False
model.generation_config.temperature = 1
model.generation_config.top_p = 1
print('disabled randomness in token prediction',
      'model generation config:', model.generation_config, sep='\n', end='\n\n')


# CREATE A PIPELINE FOR INFERENCE
try:
    print('initializing pipeline')
    pipeline = transformers.pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        model_kwargs={
            'torch_dtype': torch.float16
        })
    print('pipeline initialized')
except:
    sys.exit('Failed to initialize a pipeline')


# PREPARE CHAT TEMPLATES
# Llama uses specific chat templates
# prompts is a generator that applies the template to the inputs
print('applying chat template', end='\n')
sample = DATAFRAME.iloc[1]
example_prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a researcher rigorously screening titles and abstracts of scientific papers for inclusion or exclusion in a review paper. Use the criteria below to inform your decision. If any exclusion criteria are met or not all inclusion criteria are met, exclude the article. If all inclusion criteria are met, include the article. Type “include” or “exclude” to indicate your decision. Briefly explain your decision in one short sentence. Use the format<decision>;<explanation>. Do not type anything else. "},
        {"role": "user",
         "content": f"title: {sample['title']}\n\nabstract: {sample['abstract']}\n\ninclusion: {INCLUSION}\n\nexclusion: {EXCLUSION}"},
    ],
    tokenize=False
)
print('sample chat template:', example_prompt, sep='\n', end='\n\n')
prompts = (
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a researcher rigorously screening titles and abstracts of scientific papers for inclusion or exclusion in a review paper. Use the criteria below to inform your decision. If any exclusion criteria are met or not all inclusion criteria are met, exclude the article. If all inclusion criteria are met, include the article. Type “include” or “exclude” to indicate your decision. Briefly explain your decision in one short sentence. Use the format<decision>;<explanation>. Do not type anything else. "},
            {"role": "user",
                "content": f"title: {article['title']}\n\nabstract: {article['abstract']}\n\ninclusion: {INCLUSION}\n\nexclusion: {EXCLUSION}"},
        ],
        tokenize=False
    ) for index, article in DATAFRAME.iterrows()
)
print('chat template applied')


# This function extracts the raw response from the template
# based on an end_of_turn token
def extract_assistant_response(text, end_of_turn="<|eot_id|>assistant"):
    if end_of_turn in text:
        # add len to exclude the end_of_turn tag
        start_index = text.index(end_of_turn) + len(end_of_turn)
        return text[start_index:].strip()
    else:
        return None


# SETTING EXPORT PATH
current_time = time.time()

# Convert to a datetime object
local_time = datetime.datetime.fromtimestamp(current_time)

# Format the datetime as a string
formatted_time = local_time.strftime("%Y-%m-%d_%H:%M:%S")

EXPORT_PATH = f'./predictions/{dataset}/{formatted_time}'

if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)
    print(f'created directory {EXPORT_PATH})')

print(f'Predictions will be saved to {EXPORT_PATH}')


settings_inference = f'Settings used:\n\nDataset: {dataset}\nRows: {len(DATAFRAME) if SUBSET == 0 else SUBSET}\nBatch Size: {BATCH_SIZE}\nMax New Tokens: {MAX_NEW_TOKENS}'
settings_model = f'Model:\n{model}\n\nTokenizer:\n{tokenizer}'

settings_total = settings_inference + '\n\n' + settings_model


with open(f'{EXPORT_PATH}/settings.txt', 'w') as settings_file:
    settings_file.write(settings_total)


# DEFINE INFERENCE
counter = 0

print(settings_inference)

# log how long the inference will take
start = time.time()

# write each prediction directly to files
# the raw file contains the unformatted responses
# the result file contains decision and reason in separate columns
try:
    with open(f'{EXPORT_PATH}/{dataset}_pred_raw.csv', 'w') as raw_file, open(f'{EXPORT_PATH}/{dataset}_pred.csv', 'w', newline='') as results_file:

        raw_writer = csv.writer(raw_file)
        raw_writer.writerow(['output', 'title', 'doi', 'pubmed_id'])

        results_writer = csv.writer(results_file)
        results_writer.writerow(
            ['include', 'reason', 'title', 'doi', 'pubmed_id'])

        for output in pipeline(prompts, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, batch_size=BATCH_SIZE):
            response = extract_assistant_response(output[0]['generated_text'])

            row = DATAFRAME.iloc[counter]

            raw_writer.writerow(
                [response, row['title'], row['doi'], row['pubmed_id']])

            include, reason = response.split(';')

            include = True if include == 'include' else False
            reason.strip()  # remove whitespaces

            results_writer.writerow(
                [include, reason, row['title'], row['doi'], row['pubmed_id']])

            counter += 1

            if counter % BATCH_SIZE == 0:
                print(f"Finished processing batch {int(counter / BATCH_SIZE)}")

            del response, row, include, reason,

            torch.cuda.empty_cache()
except Exception as e:
    print(f'An error occured: {e}')
finally:
    raw_file.close()
    results_file.close()

    end = time.time()
    elapsed = end - start

    minutes = elapsed // 60
    seconds = elapsed % 60

    print(
        f'Script finished after {int(minutes)} minutes and {int(seconds)} seconds')
