"""
Various functions used in the different classes and algorithms.
"""

from enum import Enum

import pandas as pd
import streamlit as st

from helpers.data_preproccesing import run_preprocess


class SplitMethod(Enum):
    MERGED = "One merged dataset"
    SAME = "Same as uploaded"
    TRAIN_TEST = "Train / Test (80 / 20)"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Load a csv file

    :param path: path to csv file
    :return: dataframe
    """
    return pd.read_csv(path)


@st.dialog("Are you sure?")
def sumup_operations(
    operations: dict, dataset: pd.DataFrame, datasets_idx: list[int]
) -> None:
    st.markdown("### You are about to run")

    txt = "\n".join([f"- {op}" for op, _ in operations.items()])
    st.markdown(txt)

    with st.expander("export"):
        st.write(operations)

    if st.button("Confirm"):
        st.session_state["enc_dataset"] = run_preprocess(
            operations, dataset, datasets_idx
        )
        st.rerun()


def get_split_idx(
    split_method: SplitMethod,
    datasets_name: list[str],
    datasets: list[pd.DataFrame],
    dataset_idx: list[int] = None,
) -> tuple[list[int], list[str]]:
    """
    Get index to split the dataset according to the selected method.
    :param split_method: method to split the dataset
    :param datasets_name: name of the dataset(s)
    :param datasets: list of datasets
    :param dataset_idx:
    :return: list of dataset sizes and list of dataset names
    """

    datasets_idx = [len(d) for d in datasets] if dataset_idx is None else dataset_idx

    if split_method in ["One merged dataset", "Same as uploaded"]:
        if split_method == "One merged dataset":
            datasets_idx = [sum(datasets_idx)]
            datasets_name = datasets_name[:1]
        # else: keep as-is

    elif split_method == "Train / Test (80 / 20)":
        total = sum(datasets_idx)
        datasets_idx = [round(total * 0.8), round(total * 0.2)]
        datasets_name = [f"train_{datasets_name[0]}", f"test_{datasets_name[0]}"]

    return datasets_idx, datasets_name
