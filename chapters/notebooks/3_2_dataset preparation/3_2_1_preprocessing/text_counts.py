# %%


# Add length, sentence, and word count for titles and abstracts
# %%
# THIS SCRIPT CALCULATES THE LENGTH,
# SENTENCE, AND WORD COUNT FOR TITLES AND ABSTRACTS
import pandas as pd


# enable the import of custom modules
import os, sys
# # save the current working directory to direct the script to the project root
current_dir = os.getcwd()

# get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))

# add the project root to the system path
if project_root not in sys.path:
    sys.path.append(project_root)

from src.util import detect_language
from src import data

DIRECTORY = '../../../../data/03_pubmed'  # CHANGE AS NEEDED

files = [file for file in os.listdir(DIRECTORY) if file.endswith('.csv')]

datasets = data.dict_from_directory(DIRECTORY, separator=',')
# %%
for subject, dataset in datasets.items():
    datasets[subject] = data.count_vocabulary(
        dataframe=dataset, columns=['title', 'abstract'])

# %%
for index, (subject, dataset) in enumerate(datasets.items()):
    dataset.to_csv(
        f'{DIRECTORY}/{files[index]}', index=False)
