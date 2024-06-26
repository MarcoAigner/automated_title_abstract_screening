import os
import re
import pandas as pd
from typing import Dict, List, Optional, Literal


def dict_from_directory(directory: str) -> Dict[str, pd.DataFrame]:
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


def duplicates(df: pd.DataFrame, columns: Optional[str | List[str]], keep: Literal['first', 'last', False] = False) -> pd.DataFrame:
    """
    Return duplicate rows in a dataframe.

    Args:
        df (pd.DataFrame): Dataframe to check for duplicates.
        columns (str | List[str]): Columns to check for duplicates.
        keep (Literal['first', 'last', False]): Determines which duplicates to keep.

        Returns:
            pd.DataFrame: Dataframe with duplicate rows.
    """
    return df[df.duplicated(subset=columns, keep=keep)]
