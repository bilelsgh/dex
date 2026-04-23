"""
Contains classic data preprocessing function
"""

import io
import os
import zipfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


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
    logger.warning(train.columns)
    train_enc = train.drop(columns=ohe).reset_index(drop=True)

    if ohe:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        logger.debug("Encode")

        # encode cat columns
        try:
            train_cat_enc = enc.fit_transform(train[ohe])
        except:
            logger.error(
                "Error during encoding. Check that the columns to encode are of type 'object' and contain categorical values."
            )
            raise

        logger.info("ok")
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


def normalize_dataset(dataset: pd.DataFrame, col: list[str] = None) -> pd.DataFrame:
    """
    Normalize numerical values using Min-Max scaling to a range [0, 1].
    :param dataset: Dataset to normalize.
    :param col: Columns to normalize.
    :return: Normalized dataset.
    """
    logger.debug("Normalize (Min-Max)")
    # init
    df = dataset.copy()
    scaler = MinMaxScaler()

    # identify numerical columns if not provided
    num_col = col if col else df.select_dtypes(include=["number"]).columns.tolist()

    # apply scaling
    df[num_col] = scaler.fit_transform(df[num_col])

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
        "minmax": normalize_dataset,
        "encoding": encode_dataset,
        "replace_val": change_label,
        "pca": pca,
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
            # continue

        # Special handling for encoding to retrieve new columns (oh-encoded ones)
        elif operation == "encoding":
            df_dataset, encoded_columns = valid_operations[operation](
                df_dataset, **args
            )
            # continue

        else:
            # Apply all other preprocessing steps
            df_dataset = valid_operations[operation](df_dataset, **args)

    return df_dataset


def split_datasets(
    dataset: pd.DataFrame,
    indexes: list[int],
    names: list[str],
    for_download: bool = True,
    export_csv: bool = False,
) -> Union[BytesIO, Tuple[pd.DataFrame, ...]]:
    """
    Split dataset in len(indexes) subdatasets according to the given indexes.

    :param dataset: Dataset to split.
    :param indexes: Size of the subdatasets.
    :param names: Names of the subdatasets.
    :param for_download: If True, return a zip object for download, else return a tuple containing the datasets.
    :param export_csv: If True, export the datasets as csv files in the zip file (only if for_download is True).
    :return: Split dataset.
    """

    indexes_ = indexes.copy()
    indexes_.insert(0, 0)  # for the first split dataset
    datasets = [
        dataset.iloc[indexes_[i] : indexes_[i] + indexes_[i + 1]]
        for i in range(len(indexes))
    ]

    if export_csv:
        logger.debug("Exporting datasets as csv files.")
        if not os.path.isdir("preprocessed_datasets"):
            os.mkdir("preprocessed_datasets")

        for n, d in zip(names, datasets):
            d.to_csv(
                f"preprocessed_datasets/preprocessed_{os.path.basename(n)}", index=False
            )
            logger.info(
                f"Wrote {n} to preprocessed_datasets/preprocessed_{os.path.basename(n)}"
            )

    if not for_download:
        return tuple(datasets)

    # create a zip file
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "x") as csv_zip:
        for n, d in zip(names, datasets):
            csv_zip.writestr(f"preprocessed_{n}", pd.DataFrame(d).to_csv(index=False))

    return buf


def pca(
    train_df: pd.DataFrame,
    cols_to_drop: list[str],
    nb_dim_: int = None,
    min_variance: float = 0.8,
) -> pd.DataFrame:
    """
    Reducing dimension using PCA.

    :param train_df: Train dataframe.
    :param cols_to_drop: Columns to drop before applying PCA (label columns, attack_cat, etc.)
    :param nb_dim_: Number of dimensions to keep. If None, determined by min_variance.
    :param min_variance: Minimum cumulated variance to reach when selecting nb_dim.

    :return reduced_train_df: Train dataframe with reduced dimensions
    :return fitted_pca: Fitted PCA object
    """

    train_features = train_df.drop(columns=cols_to_drop)
    train_dropped = train_df[cols_to_drop]

    # Determine number of dimensions to keep
    if nb_dim_ is not None:
        nb_dim = nb_dim_
    else:
        full_pca = PCA().fit(train_features)
        cumulative_variances = np.cumsum(full_pca.explained_variance_ratio_)
        nb_dim = next(
            (i + 1 for i, v in enumerate(cumulative_variances) if v >= min_variance),
            train_features.shape[1],  # fallback: keep all dimensions
        )

    # Fit PCA with the selected number of dimensions on train, apply to both sets
    fitted_pca = PCA(n_components=nb_dim)
    fitted_pca.fit(train_features)

    reduced_train_df = pd.DataFrame(fitted_pca.transform(train_features))

    # Add back label columns
    reduced_train_df[cols_to_drop] = train_dropped

    # viz pca
    pca_component_directions = pd.DataFrame(
        fitted_pca.components_,
        columns=train_features.columns,
        index=np.arange(1, fitted_pca.n_components_ + 1),
    )

    # Make a heatmap to show the contribution of each feature to each principal component
    fig = plt.figure(figsize=(12, 9))
    sns.heatmap(
        pca_component_directions.T,
        linewidth=0.2,
        annot=False,
        cmap="coolwarm",
        vmax=1,
        vmin=-1,
    )
    plt.ylabel("Features", fontsize=11)
    plt.xlabel("Components", fontsize=11)
    plt.tight_layout()
    plt.savefig("exp/pca_component_directions.png", dpi=300)

    return reduced_train_df
