import os
import re
import pandas as pd
from typing import Dict


def dataset_dict_from_directory(directory: str) -> Dict[str, pd.DataFrame]:
    """
    Return a dictionary containing dataframes from all .csv-files in a directory.

    Args:
        directory (str): Path to directory containing .csv-files.

    Returns:
        dict: Dictionary with subjects as keys and dataframes as values.
    """

    # all .csv-files in the directory
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # expects files to end with '_[annotation].csv'
    pattern = r'^(.*)_.*$'

    # extract subjects from filenames
    subjects = [re.search(pattern, file).group(1) for file in files]

    # return dictionary with subjects as keys and dataframes as values
    return {
        subjects[count]: pd.read_csv(
            f'{directory}/{file}').convert_dtypes()
        for count, file in enumerate(files)
    }
