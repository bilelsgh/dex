"""
Various functions used in the different classes and algorithms.
"""

import streamlit as st
import pandas as pd

from helpers.data_preproccesing import run_preprocess

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
