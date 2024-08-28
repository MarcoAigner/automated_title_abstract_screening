# %%


# Add length, sentence, and word count for titles and abstracts
# %%
# THIS SCRIPT CALCULATES THE LENGTH,
# SENTENCE, AND WORD COUNT FOR TITLES AND ABSTRACTS
import pandas as pd
from src import data
import os

DIRECTORY = '../../../../data/datasets/03_pubmed'  # CHANGE AS NEEDED

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
