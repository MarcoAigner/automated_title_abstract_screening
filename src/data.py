import os
import re
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Literal
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk


def dict_from_directory(directory: str, separator: Optional[str] = ',', type: Literal['pandas', 'polars'] = 'pandas', with_index: Optional[bool] = False) -> Dict[str, pd.DataFrame | pl.DataFrame]:
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

    # whether to include index in dataframes
    index_value = False if not with_index else 'index'

    if type == 'pandas':
        # return dictionary with subjects as keys and dataframes as values
        return {
            subjects[count]: pd.read_csv(
                f'{directory}/{file}',
                sep=separator,
                index_col=index_value
            ).convert_dtypes()
            for count, file in enumerate(files)
        }
    elif type == 'polars':
        return {
            subjects[count]: pl.read_csv(
                f'{directory}/{file}',
                separator=separator,
                row_index_name=index_value
            )
            for count, file in enumerate(files)
        }


def duplicates(df: pd.DataFrame, columns: Optional[str | List[str]], keep: Literal['first', 'last', False] = False) -> pd.DataFrame:
    """
    Return duplicate rows in a dataframe.

    Args:
        df (pd.DataFrame): Dataframe to check for duplicates.
        columns (str | List[str]): Columns to check for duplicates.
        keep (Literal['first', 'last', False]): Determines which duplicates to keep. False keeps all duplicates.

    Returns:
        pd.DataFrame: Dataframe with duplicate rows.
    """
    return df[df.duplicated(subset=columns, keep=keep)]


def column_length(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate the length of the column

    Args:
        dataframe (pd.DataFrame): Dataframe to calculate the length of the column
        column (str): Column to calculate the length of

    Returns:
        pd.DataFrame: Dataframe with the length of the column
    """
    return dataframe[column].apply(lambda x: len(x) if pd.notnull(x) else 0)


def word_counts(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate the number of words in the column

    Args:
        dataframe (pd.DataFrame): Dataframe to calculate the number of words in the column
        column (str): Column to calculate the number of words in

    Returns:
        pd.DataFrame: Dataframe with the number of words in the column
    """
    return dataframe[column].apply(lambda x: len(word_tokenize(x)) if pd.notnull(x) else 0)


def sentence_counts(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate the number of sentences in the column

    Args:
        dataframe (pd.DataFrame): Dataframe to calculate the number of sentences in the column
        column (str): Column to calculate the number of sentences in

    Returns:
        pd.DataFrame: Dataframe with the number of sentences in the column
    """
    return dataframe[column].apply(lambda x: len(sent_tokenize(x)) if pd.notnull(x) else 0)


def count_vocabulary(dataframe: pd.DataFrame, columns: List[str], length: bool = True, count_words: bool = True, count_sentences: bool = True) -> pd.DataFrame:
    """
    Count lengt, words and sentences in columns of a dataframe.

    Args:
        dataframe (pd.DataFrame): Dataframe to count vocabulary in.
        columns (List[str]): Columns to count vocabulary in.
        length (bool): Whether to count length of columns.
        count_words (bool): Whether to count words in columns.
        count_sentences (bool): Whether to count sentences in columns.

    Returns:
        pd.DataFrame: Dataframe with vocabulary counts.
    """

    # download punkt tokenizer from the natural language toolkit
    nltk.download('punkt_tab')

    for column in columns:
        if length:
            dataframe[f'{column}_length'] = column_length(dataframe, column)
        if count_words:
            dataframe[f'{column}_word_count'] = word_counts(dataframe, column)
        if count_sentences:
            dataframe[f'{column}_sentence_count'] = sentence_counts(
                dataframe, column)

    return dataframe
