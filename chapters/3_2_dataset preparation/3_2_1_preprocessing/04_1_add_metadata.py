# THIS SCRIPT CALCULATES THE LENGTH,
# SENTENCE, AND WORD COUNT FOR TITLES AND ABSTRACTS
# AND APPENDS THESE DATA TO THE PROVIDED FILES
# ALSO DETECTS THE LANGUAGE OF THE TITLE AND ABSTRACT
# RUN BEFORE THE PREPROCESSING NOTEBOOK

# %%
# 1. ADD CUSTOM MODULES TO PATH
import sys, os

# recursively search for the root directory containing a specific file
def find_root_dir(search_for='.gitignore'):

    current_dir = os.getcwd()

    while True:
        if os.path.exists(os.path.join(current_dir, search_for)):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(
                f"Could not find '{search_for}' in any parent directory.")
        current_dir = parent_dir


# save the root directory to a variable
root_dir = find_root_dir()
print(f"Root directory found: {root_dir}")

# add the root directory to the system path
sys.path.append(root_dir)
if root_dir in sys.path:
    print(f"Root directory added to system path.")

# %%
# 2. IMPORT THE DATASETS
from src import data

# where the datasets are stored
DIRECTORY = '../../../data/datasets/03_pubmed'

# list all csv files in the directory
files = [file for file in os.listdir(DIRECTORY) if file.endswith('.csv')]

# create the datasets dictionary
datasets = data.dict_from_directory(
    directory=DIRECTORY,
    type='polars'
)

print(f"Found {len(datasets)} datasets in {DIRECTORY}:")
print(*datasets.keys(), sep='\n')

# %%
# 3. ADD METADATA: COUNTS & LANGUAGE
import polars as pl
from src.util import detect_language
from typing import cast

# iterate over the datasets
for subject, dataset in datasets.items():

    # Ensure dataset is a polars DataFrame with explicit casting
    dataset = cast(pl.DataFrame, dataset)

    # add vocabulary counts to dataframe
    dataset = data.count_vocabulary(
        dataframe=dataset,
        columns=['title', 'abstract']
    )

    # Ensure it's still a polars DataFrame after count_vocabulry
    dataset = cast(pl.DataFrame, dataset)

    # add language of title & abstract to dataframe
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
# 4. WRITE THE DATASETS BACK TO CSV
from typing import cast 
for index, (subject, dataset) in enumerate(datasets.items()):

    # Ensure Pylance knows this is a DataFrame
    dataset = cast(pl.DataFrame, dataset)
    
    dataset.write_csv(f'{DIRECTORY}/{files[index]}')
    # dataset.write_csv(
    #     f'{DIRECTORY}/{files[index]}')

# %%
