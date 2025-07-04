import os
import re
import nltk
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Literal
from nltk.tokenize import word_tokenize, sent_tokenize


def dict_from_directory(
        directory: str,
        separator: str = ',',
        type: Literal['polars', 'pandas'] = 'polars',
        with_index: Optional[bool] = False
) -> Dict[str, pl.DataFrame | pd.DataFrame]:
    """
    Return a dictionary containing dataframes from all .csv-files in a directory.

    Args:
        directory (str): Path to directory containing .csv-files.
        separator (str): Separator used in the .csv-files.
        type (Literal['polars', 'pandas']): Whether to return a polars or pandas dataframe.
        with_index (Optional[bool]): Whether to use the first column as index. Defaults to False.,

    Returns:
        dict: Dictionary with subjects as keys and dataframes as values.
    """

    # create a list of all .csv-files within the directory
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # this regex pattern searches for the format of _[annotation].csv'
    pattern = r'^(.*)_.*$'

    # extract subjects from filenames
    subjects = []
    for file in files:  # search each file for the pattern
        match = re.search(pattern, file)
        if match is not None:  # subject found, append it to the list
            subjects.append(match.group(1))
        else:
            continue  # no subject found, continue with next file

    # handle polars and pandas dataframes differently
    if type == 'pandas':
        return {
            subjects[count]: pd.read_csv(
                f'{directory}/{file}',
                sep=separator,
                index_col=False if not with_index else 'index'
            ).convert_dtypes()
            for count, file in enumerate(files)
        }
    elif type == 'polars':
        return {
            subjects[count]: pl.read_csv(
                f'{directory}/{file}',
                separator=separator,
                row_index_name=None if not with_index else 'index'
            )
            for count, file in enumerate(files)
        }

# TODO: Check if this function is used anywhere, if not remove it


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


def count_vocabulary(
        dataframe: pl.DataFrame,
        columns: List[str],
        length: bool = True,
        count_words: bool = True,
        count_sentences: bool = True
) -> pl.DataFrame:
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

    # compute the length, word and sentence counts for each specified column
    for column in columns:
        if length:
            dataframe = dataframe.with_columns(
                pl.col(column)
                .map_elements(
                    function=lambda x: column_length(
                        dataframe, column) if x is not None else 0,
                    return_dtype=pl.Int64
                )
            )
        if count_words:
            dataframe = dataframe.with_columns(
                pl.col(column)
                .map_elements(
                    function=lambda x: word_counts(
                        dataframe, column) if x is not None else 0,
                    return_dtype=pl.Int64
                )
            )
        if count_sentences:
            dataframe[f'{column}_sentence_count'] = sentence_counts(
                dataframe, column)

    return dataframe


def column_length(dataframe: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Calculate the length of the column

    Args:
        dataframe (pl.DataFrame): Dataframe to calculate the length of the column
        column (str): Column to calculate the length of

    Returns:
        pl.DataFrame: Dataframe with the length of the column
    """

    return dataframe.with_columns(
        pl.col(column).str.len_chars().alias(f'{column}_length')
    )


def word_counts(dataframe: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Calculate the number of words in the column

    Args:
        dataframe (pd.DataFrame): Dataframe to calculate the number of words in the column
        column (str): Column to calculate the number of words in

    Returns:
        pd.DataFrame: Dataframe with the number of words in the column
    """

    return dataframe.with_columns(
        pl.col(column)
        .map_elements(
            function=lambda x: len(word_tokenize(x)) if x is not None else 0,
            return_dtype=pl.Int64
        ).alias(f'{column}_word_count')
    )
    # return dataframe[column].apply(lambda x: len(word_tokenize(x)) if pd.notnull(x) else 0)


def sentence_counts(dataframe: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Calculate the number of sentences in the column

    Args:
        dataframe (pd.DataFrame): Dataframe to calculate the number of sentences in the column
        column (str): Column to calculate the number of sentences in

    Returns:
        pd.DataFrame: Dataframe with the number of sentences in the column
    """

    return dataframe.with_columns(
        pl.col(column)
        .map_elements(
            function=lambda x: len(sent_tokenize(x)) if x is not None else 0,
            return_dtype=pl.Int64
        ).alias(f'{column}_sentence_count')
    )
