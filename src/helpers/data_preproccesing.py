"""
Contains classic data preprocessing function
"""

import io
import zipfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def encode_dataset(
    train: pd.DataFrame,
    ohe: List[str] = [],
    le: List[str] = [],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Encode categorical columns into numerical columns.
    LabelEncoder will encode label columns into numerical columns.

    :param train: Train dataset.
    :param ohe: Column to Ohe Hot Encode
    :param le: Column to encode with Label Encoder
    :return: Encoded datasets
    :return: New columns from ohe
    """

    new_columns = {}
    train_enc = train.drop(columns=ohe).reset_index(drop=True)

    if ohe :
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        logger.debug("Encode")

        # encode cat columns
        train_cat_enc = enc.fit_transform(train[ohe])

        encoded_columns = enc.get_feature_names_out(ohe)  # new columns name
        train_enc = pd.concat(
            [
                train_enc,
                pd.DataFrame(train_cat_enc, columns=encoded_columns),
            ],
            axis=1,
        )
        new_columns = {
            feat: [f"{feat}_{cat}" for cat in cats]
            for feat, cats in zip(enc.feature_names_in_, enc.categories_)
        }

    # encode label
    for c in le:
        l_enc = LabelEncoder()
        train_enc[c] = l_enc.fit_transform(train[c])

    return train_enc, new_columns


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


def change_label(dataset: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Change label column values.

    :param dataset: Dataset to change.
    :param mappings: Dict containing the mappings {
        <column_name>: [
                            {
                                "former_values": ["...", ..],
                                "new_value": ".."
                            }
                    ]
        ...
    }
    :return:
    """

    df = dataset.copy()
    logger.debug(f"df length: {len(df)}")
    # iterate over columns
    for col_name, values in mappings.items():

        # iterate over mappings
        for mapping in values:
            df[col_name] = df[col_name].replace(
                mapping["former_values"], mapping["new_value"]
            )

        df[col_name] = df[col_name].astype("string")

    logger.debug(f"df length: {len(df)}")
    return df


def run_preprocess(
    operations: dict[str, dict], dataset: pd.DataFrame, datasets_idx: list[int]
) -> pd.DataFrame:
    """
    Run preprocessing operations on a dataset.

    :param operations: Dict mapping operation names to their parameters.
    :param dataset: Dataset to preprocess.
    :param datasets_idx: List of dataset split sizes (used for standardization).
    :return: Preprocessed dataset.
    """

    # Map operation names to their corresponding functions
    valid_operations = {
        "remove_inv_val": remove_invalid_val,
        "standardization": standardize_dataset,
        "encoding": encode_dataset,
        "replace_val": change_label,
    }

    # Work on a copy to avoid modifying the original dataset
    df_dataset = dataset.copy()
    encoded_columns: Optional[
        Dict[str, list[str]]
    ] = None  # to keep track of the new columns generated by encoding

    total_ops = len(operations)
    progress = st.progress(0, "Operation in progress..")

    # Prepare sizes for standardization step (avoid data leakage)
    idxes = datasets_idx.copy()
    idxes.insert(0, 0)  # first split always starts at 0

    for idx, (operation, args) in enumerate(operations.items()):
        # Update progress bar
        progress.progress((idx + 1) / total_ops)

        # Check for unknown operations
        if operation not in valid_operations:
            raise ValueError(f"Unknown operation: {operation}")

        # Special handling for standardization to avoid leakage between splits
        if operation == "standardization":
            logger.info("About to split dataset before standardization.")

            # split dataset to separate standardization operations
            datasets = [
                df_dataset.iloc[idxes[i] : idxes[i] + idxes[i + 1]]
                for i in range(len(datasets_idx))
            ]

            # add new columns if needed
            args["col"] = [
                new_c
                for c in args["col"]
                for new_c in (encoded_columns[c] if c in encoded_columns else [c])
            ]

            # standardize
            df_dataset = pd.concat(
                [valid_operations[operation](d, **args) for d in datasets],
                ignore_index=True,
            )
            continue

        # Special handling for encoding to retrieve new columns (oh-encoded ones)
        elif operation == "encoding":
            df_dataset, encoded_columns = valid_operations[operation](
                df_dataset, **args
            )
            continue

        # Apply all other preprocessing steps
        df_dataset = valid_operations[operation](df_dataset, **args)

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
    indexes_.insert(0, 0)  # for the first split dataset
    datasets = [
        dataset.iloc[indexes_[i] : indexes_[i] + indexes_[i + 1]]
        for i in range(len(indexes))
    ]

    if not for_download:
        return tuple(datasets)

    # create a zip file
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "x") as csv_zip:
        for n, d in zip(names, datasets):
            csv_zip.writestr(f"preprocessed_{n}", pd.DataFrame(d).to_csv(index=False))

    return buf
