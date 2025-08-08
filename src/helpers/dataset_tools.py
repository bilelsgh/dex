"""
Class and functions to ease the use and manipulation of datasets
"""
import sys
from typing import Union

import pandas as pd
import streamlit as st

from helpers.data_preproccesing import split_datasets

sys.path.append("../..")


class DatasetDownloader:
    """
    Help to have the good configuration for dataset(s) downloading
    """

    def __init__(
        self,
        datasets_idx: list[int],
        processed_dataset: pd.DataFrame,
        datasets_names: Union[list[str], str],
    ):
        """

        :param datasets_idx: Size of the datasets
        :param processed_dataset: The processed dataset
        :param datasets_names: Name of every dataset
        """
        self.datasets_idx = datasets_idx
        self.processed_dataset = processed_dataset
        self.datasets_name = datasets_names
        self.is_single = len(datasets_idx) <= 1

    @property
    def download_file(self):
        return (
            st.session_state["enc_dataset"].to_csv(index=False)
            if self.is_single
            else split_datasets(
                self.processed_dataset, self.datasets_idx, self.datasets_name
            )
        )

    @property
    def file_name(self):
        return (
            f"processed_{self.datasets_name[0].split('.')[0]}.csv"
            if self.is_single
            else "preprocessed_dataset.zip"
        )

    @property
    def mime_type(self):
        return "text/csv" if self.is_single else "application/zip"
