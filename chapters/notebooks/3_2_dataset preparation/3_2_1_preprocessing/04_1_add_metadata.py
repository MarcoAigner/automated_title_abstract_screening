# %%


# Add length, sentence, and word count for titles and abstracts
# %%
# THIS SCRIPT CALCULATES THE LENGTH,
# SENTENCE, AND WORD COUNT FOR TITLES AND ABSTRACTS
# AND APPENDS THESE DATA TO THE PROVIDED FILES
# ALSO DETECTS THE LANGUAGE OF THE TITLE AND ABSTRACT
# RUN BEFORE THE PREPROCESSING NOTEBOOK
import pandas as pd
import polars as pl
from src.util import detect_language
from src import data
import os

DIRECTORY = '../../../../data/datasets/03_pubmed'  # CHANGE AS NEEDED

files = [file for file in os.listdir(DIRECTORY) if file.endswith('.csv')]

datasets = data.dict_from_directory(DIRECTORY, separator=',')
# %%
for subject, dataset in datasets.items():

    # add vocabulary counts to dataframe
    dataset = data.count_vocabulary(
        dataframe=dataset,
        columns=['title', 'abstract']
    )

    # add language of title & abstract to dataframe
    # requires polars
    dataset = pl.DataFrame(dataset).with_columns(
        pl.col('title').map_elements(
            lambda x: detect_language(x),
            return_dtype=pl.String
        ).alias('language_title'),
        pl.col('abstract').map_elements(
            lambda x: detect_language(x),
            return_dtype=pl.String
        ).alias('language_abstract')
    )

    # override the dataset in the collection
    datasets[subject] = dataset

# %%
for index, (subject, dataset) in enumerate(datasets.items()):
    dataset.write_csv(
        f'{DIRECTORY}/{files[index]}')
