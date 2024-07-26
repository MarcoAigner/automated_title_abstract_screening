# %%
import pandas as pd
from src import data
import os

# Add length, sentence, and word count for titles and abstracts
# %%
data_directory_pubmed = '../../../../data/04_pubmed'

datasets = data.dict_from_directory(data_directory_pubmed, separator=',')
# %%
for subject, dataset in datasets.items():
    datasets[subject] = data.count_vocabulary(
        dataframe=dataset, columns=['title', 'abstract'])

# %%
data_directory_vocabs = '../../../../data/05_vocabs'

for subject, dataset in datasets.items():
    dataset.to_csv(
        f'{data_directory_vocabs}/{subject}_vocabs.csv', index=False)
