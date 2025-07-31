"""
Contains classic data preprocessing function
"""

import io
import zipfile
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def encode_dataset(
    train: pd.DataFrame,
    ohe: List[str] = [],
    le: List[str] = [],
) -> pd.DataFrame:
    """
    Encode categorical columns into numerical columns.
    LabelEncoder will encode label columns into numerical columns.

    :param train: Train dataset.
    :param ohe: Column to Ohe Hot Encode
    :param le: Column to encode with Label Encoder
    :return: Encoded datasets
    """

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    logger.debug("Encode")

    # encode cat columns
    train_cat_enc = enc.fit_transform(train[ohe])

    encoded_columns = enc.get_feature_names_out(ohe)  # new columns name
    train_enc = pd.concat(
        [
            train.drop(columns=ohe).reset_index(drop=True),
            pd.DataFrame(train_cat_enc, columns=encoded_columns),
        ],
        axis=1,
    )

    # encode label
    for c in le:
        l_enc = LabelEncoder()
        train_enc[c] = l_enc.fit_transform(train[c])

    return train_enc


def standardize_dataset(dataset: pd.DataFrame, col: list[str] = None) -> pd.DataFrame:
    """
    Standardize numerical values by subtracting the mean and dividing by the standard deviation.

    :param dataset: Dataset to standardize.
    :param col: Columns to standardize.
    :return: Standardized dataset.
    """

    logger.debug("Standardize")
    # init
    df = dataset.copy()
    std = StandardScaler()
    num_col = (
        col if col else df.select_dtypes(include=["number"]).columns.tolist()
    )  # keep numerical columns

    # std
    df[num_col] = std.fit_transform(df[num_col])

    return df


def remove_invalid_val(dataset: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Handle the following invalid values:
        - Infinite values are replaced by k times the max/min value and Nan
        - NaN values are replaced by the mean of the column

    :param dataset: Dataset to change.
    :param k: The inf values will be replaced by k * times the max / min value - default=10
    :return:Dataframe without inf values
    """

    df = dataset.copy()

    for col in df.columns:
        if (
            df[col].dtype.kind in "bifc"
        ):  # We go through column containing numeric values

            # Replace -inf with -k * abs max value
            max_val = df.loc[~df[col].isin([np.inf, -np.inf]), col].abs().max()
            df.loc[df[col] == -np.inf, col] = -k * max_val

            # Replace inf with k * max value
            df.loc[df[col] == np.inf, col] = k * max_val

            # For numeric columns, Nan are replaced by the mean
            replaced_nan_val = df.loc[
                ~df[col].isin([np.inf, -np.inf, np.nan]), col
            ].mean()

        else:
            # For obj columns, Nan are replaced by the mode
            replaced_nan_val = df[col].mode()[0]

        df.loc[df[col].isna(), col] = replaced_nan_val

    return df


def run_preprocess(
    operations: dict[str, list[str]], dataset: pd.DataFrame
) -> pd.DataFrame:
    """
    Run preprocessing operations.

    :param operations: Desired preprocessing operations and args.
    :param dataset: Dataset to preprocess.
    :return:
    """

    valid_operations = {
        "remove_inv_val": remove_invalid_val,
        "standardization": standardize_dataset,
        "encoding": encode_dataset,
    }
    df_dataset = dataset.copy()
    progress = st.progress(0, "Operation in progress..")

    for idx, obj in enumerate(operations.items()):
        operation, args = obj
        progress.progress((1 + idx) * 1 / len(operations))

        # try:
        df_dataset = valid_operations[operation](df_dataset, **args)
        # except KeyError as e:
        #     raise ValueError(f"{e} is not a valid operation")

    return df_dataset


def split_datasets(
    dataset: pd.DataFrame,
    indexes: list[int],
    names: list[str],
    for_download: bool = True,
) -> Union[BytesIO, Tuple[pd.DataFrame, ...]]:
    """
    Split dataset in len(indexes) subdatasets according to the given indexes.

    :param dataset: Dataset to split.
    :param indexes: Size of the subdatasets.
    :param names: Names of the subdatasets.
    :param for_download: If True, return a zip object for download, else return a tuple containing the datasets.
    :return: Split dataset.
    """

    indexes_ = indexes.copy()
    indexes_.insert(0, -1)  # for the first split dataset
    datasets = [
        dataset.iloc[indexes_[i] + 1 : indexes_[i + 1]] for i in range(len(indexes))
    ]

    if not for_download:
        return tuple(datasets)

    # create a zip file
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "x") as csv_zip:
        for n, d in zip(names, datasets):
            csv_zip.writestr(f"preprocessed_{n}", pd.DataFrame(d).to_csv())

    return buf
